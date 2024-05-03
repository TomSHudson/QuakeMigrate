# -*- coding: utf-8 -*-
"""
Module for reading in das data

:copyright:
    2020–2024, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from itertools import chain
import logging
import pathlib
import glob 
import numpy as np 
from scipy import ndimage, signal, integrate
from obspy import read, Stream, Trace, UTCDateTime
import quakemigrate.util as util
import quakemigrate.io.dasio.load_das_h5 as load_das_h5


def read_das(das_archive_path, das_data_fmt, starttime, endtime, pre_pad=0.0, post_pad=0.0, 
            first_last_das_channels=[0,-1], duplicate_das_comps=True, station_prefix="D", 
            spatial_down_samp_factor=1, fk_filter_params={}, apply_notch_filter=False, 
            notch_freqs=[], notch_bw=2.5, semblance_stack=False, semblance_v_app_min=1.0,
            convert_strainrate_to_vel=False, channel_spacing=None, gauge_length=None):
    """
    Read in das data for a particular time period, and output to obspy stream.

    Parameters
    ----------
    das_archive_path : `pathlib.Path` object
        If das data is also to be included in analysis, ths is the path to the das data 
        archive. This currently has to be of a somewhat specific format, where all das 
        files are in the same directory and are labelled by time. Currently, only files 
        with start time in the format: *UTC_YYYYMMDD_HHMMSS.???.<das_data_fmt> 
        are supported. <das_data_fmt> is specified as another attribute of Archive. 
        Default is das_archive_path = None, resulting in no das data being included in 
        the analysis.
    das_data_fmt : str
        If das data is included (i.e. if <das_archive_path> is specified), then this is 
        the das format to be read. Currently, the only supported formats are h5 and sgy, 
        but it is relatively trivial to support other formats in the future (contact the 
        developers or fork repository). Default is h5.
    starttime : `obspy.UTCDateTime` object
        Timestamp from which to read waveform data.
    endtime : `obspy.UTCDateTime` object
        Timestamp up to which to read waveform data.
    pre_pad : float, optional
        Additional pre pad of data to read. Defaults to 0.
    post_pad : float, optional
        Additional post pad of data to read. Defaults to 0.
    first_last_das_channels : list of 2x ints, optional
        If specified, selects only certain DAS channels along the fibre, from 
        first_last_das_channels[0] to first_last_das_channels[1]. Default is to use 
        all channels.
    duplicate_das_comps : bool, optional
        If duplicate_das_comps is True, will set das data for each channel as EHZ, EHN 
        and EHE. Otherwise, will set real das data on channel EHN and not allocate other 
        channels.
    station_prefix : str, optional
        Station name prefix for das channels. das channels are then named with the 
        station prefix followed by an integer representing the distance along the fibre.
        Default is "D".
    spatial_down_samp_factor : int, optional
        If specified, will downsample das data spatially by the factor. 
    fk_filter_params : dict, optional
        If specified, will apply an fk filter to the data. Applied here as most efficient 
        to do it before the 2D data is split.
        keys are: "wavenumber", "max_freq" and "v_app_filts". First two are floats, 
        corresponding to the max. wavenumber and max. freq. to pass, respectively, 
        and v_app_filts is None or a list of floats, corresponding to specific apparent 
        velocities to remove (e.g. due to continuous noise from a single source). If in 
        doubt, set fk_filter_params["v_app_filts"] = None.
        Default is to not apply a fk filter.
    apply_notch_filter : bool, optional
        If True, applies a notch filter, typically applied to remove generator noise. 
        Default is False. If specified, should also specify notch_freqs (=[]) and  
        notch_bw (=2.5).
    semblance_stack : bool, optional
        If True and das data is to be spatially downsampled, then will downsample das data, 
        but stacking using semblance based stacking. Default is not to perform semblance 
        stacking, as not particularly computationally efficient.
    semblance_v_app_min : float, optional
        Only used if semblance_stack=True. This is the minimum apparent velocity to be expected 
        for a plane wave arriving at the fibre, in units of km/s. Typically, one might set this 
        to the minimum S-wave velocity expected. Default is 1 km/s.
    convert_strainrate_to_vel : bool, optional
        If True, will convert das strain-rate data to velocity. Note, that if strain data is 
        passed, then will instead convert to displacement. Default is False.
    channel_spacing : float, optional
        Used if data format is SEGY (das_data_fmt = sgy). Channel spacing of DAS data in metres.
        Default is None. Must be specified if data format is SEGY.
    gauge_length : float, optional
        Used if data format is SEGY (das_data_fmt = sgy). Gauge length of DAS data in metres.
        Default is None. Must be specified if data format is SEGY.

    Returns
    -------
    out : st
        obspy stream containing das data (1 trace per channel).

    """
    # Find files matching time window from das archive:
    das_fnames = sorted(glob.glob(str(pathlib.Path(das_archive_path, "*."+das_data_fmt))))
    abs_time_diffs = []
    for das_fname in das_fnames:
        das_f_starttime = _get_das_starttime_from_fname(das_fname, das_data_fmt)
        abs_time_diffs.append(np.abs((starttime - pre_pad) - das_f_starttime))
    abs_time_diffs = np.array(abs_time_diffs)
    nearest_idx = np.argmin(abs_time_diffs)
    das_f_starttime = _get_das_starttime_from_fname(das_fnames[nearest_idx], das_data_fmt)
    das_f_dur_s = _get_das_starttime_from_fname(das_fnames[nearest_idx], das_data_fmt) - _get_das_starttime_from_fname(das_fnames[nearest_idx-1], das_data_fmt)

    # Do a check on window vs das file properties:
    if (endtime - starttime) >  das_f_dur_s:
        print("Warning: window length (s) > than das file duration, which may raise error.")

    # Append files to read:
    das_fnames_to_read = []
    # If window start is after best fit file:
    if (starttime - pre_pad) - das_f_starttime >= 0:
        das_fnames_to_read.append( das_fnames[np.argmin(abs_time_diffs)] )
        if (endtime + post_pad) - das_f_starttime > das_f_dur_s:
             das_fnames_to_read.append( das_fnames[np.argmin(abs_time_diffs)+1] )
    # Else if it is before best fit file:
    else:
        das_fnames_to_read.append( das_fnames[np.argmin(abs_time_diffs)] )
        das_fnames_to_read.append( das_fnames[np.argmin(abs_time_diffs) - 1] )

    # And read in streams for each das file:
    st = Stream()
    for das_fname in das_fnames_to_read:
        if das_data_fmt == "h5":
            st += read_das_h5(das_fname, first_last_das_channels=first_last_das_channels, station_prefix="D", 
                        spatial_down_samp_factor=spatial_down_samp_factor, fk_filter_params=fk_filter_params, 
                        duplicate_Z_and_E=duplicate_das_comps, apply_notch_filter=apply_notch_filter, 
                        notch_freqs=notch_freqs, notch_bw=notch_bw, semblance_stack=semblance_stack, 
                        semblance_v_app_min=semblance_stack, convert_strainrate_to_vel=convert_strainrate_to_vel)
        elif das_data_fmt == "sgy":
            st += read_das_segy(das_fname, first_last_das_channels=first_last_das_channels, station_prefix="D", 
                        spatial_down_samp_factor=spatial_down_samp_factor, fk_filter_params=fk_filter_params, 
                        duplicate_Z_and_E=duplicate_das_comps, apply_notch_filter=apply_notch_filter, 
                        notch_freqs=notch_freqs, notch_bw=notch_bw, semblance_stack=semblance_stack, 
                        semblance_v_app_min=semblance_stack, convert_strainrate_to_vel=convert_strainrate_to_vel,
                        channel_spacing=channel_spacing, gauge_length=gauge_length)
    st = util.merge_stream(st)

    return st


def _get_das_starttime_from_fname(das_fname, das_data_fmt):
    """Function to get das starttime as UTCDateTime object from path object."""
    das_fname_tmp = str(pathlib.PurePath(das_fname).parts[-1])
    str_tmp = das_fname_tmp.split("."+das_data_fmt)[0]
    if das_data_fmt == "h5":
        das_f_starttime_str = str_tmp.split("UTC_")[-1]
        das_f_starttime = UTCDateTime(year=int(das_f_starttime_str[0:4]), 
                                    month=int(das_f_starttime_str[4:6]),
                                    day=int(das_f_starttime_str[6:8]),
                                    hour=int(das_f_starttime_str[9:11]),
                                    minute=int(das_f_starttime_str[11:13]),
                                    second=int(das_f_starttime_str[13:15]),
                                    microsecond=int((10**6) * (10**(-1 * len(das_f_starttime_str[16:]))) 
                                                * int(das_f_starttime_str[16:])))
    elif das_data_fmt == "sgy":
        str_tmp = str_tmp.split("_UTC")[0]
        str_tmp = str_tmp.split("decimator_")[-1]
        str_tmp = str_tmp.replace("_", "T")
        das_f_starttime_str = str_tmp.replace(".", ":")
        das_f_starttime = UTCDateTime(das_f_starttime_str)
    else:
        print("Error: <das_data_fmt> =", das_data_fmt, "not supported.")
    return das_f_starttime


def read_das_h5(das_fname, first_last_das_channels=[0,-1], network_code="AA", station_prefix="D", 
                spatial_down_samp_factor=1, fk_filter_params={}, duplicate_Z_and_E=True, 
                apply_notch_filter=False, notch_freqs=[], notch_bw=2.5, semblance_stack=False,
                semblance_v_app_min=1.0, convert_strainrate_to_vel=False):
    """Function to read in single das h5 file and output as obspy stream object."""
    # Get start time of data:
    data_and_headers = load_das_h5.load_file(das_fname)
    data = data_and_headers[0]
    headers = data_and_headers[1]
    das_start = UTCDateTime(headers['t0'])

    # Create station labels:
    # (based on distance along fibre)
    das_station_labels = []
    das_station_idxs = []
    end_channel = first_last_das_channels[1] 
    if end_channel == -1:
        end_channel = data.shape[1]
    for i in np.arange(first_last_das_channels[0], end_channel, int(spatial_down_samp_factor), dtype=int):
        dist_along_fibre_curr = int(np.round(data_and_headers[2]['dd'][i]))
        das_station_labels.append(''.join((station_prefix, str(dist_along_fibre_curr).zfill(4))))
        das_station_idxs.append(i)

    # And process data:
    starttime_curr = das_start
    fs = float(data_and_headers[1]['fs'])
    channel_spacing = data_and_headers[1]['dx']
    n_samp = len(data[:,0])
    endtime_curr = starttime_curr + (n_samp / fs)

    # Filter data:
    # Apply fk filter:
    if len(list(fk_filter_params.keys())) > 0:
        print('Applying fk filter')
        data = fk_filter(data, fs, channel_spacing, fk_filter_params['wavenumber'], fk_filter_params['max_freq'], 
                        v_app_filts=fk_filter_params['v_app_filts'])
    # Apply notch filter:
    if apply_notch_filter:
        print('Applying notch filter/s')
        for f_notch in notch_freqs:
            data = notch_filter(data, fs, f_notch, notch_bw, filt_axis=0)

    # Convert data to velocity, if specified:
    if convert_strainrate_to_vel:
        print("Converting das strain-rate to velocity.")
        data = strainrate2vel(data, headers)
    
    # Perform semblance stack, if decimating data and semblance stacking is specified:
    if not spatial_down_samp_factor==1:
        if semblance_stack:
            win_len = int(0.5*fs) # Set window length to 1/2 a second
            max_inter_ch_t_shift = int(np.ceil(channel_spacing / (semblance_v_app_min * 1000))) # (This should be based 
            # on slowest apparent velocity, i.e. dx/v_app, in samples)
            data = semblance_stack_all(data, win_len, spatial_down_samp_factor, max_inter_ch_t_shift=max_inter_ch_t_shift)
    
    # Loop over das channels to save:
    st = Stream()
    for i in range(len(das_station_labels)):
        # Add data to stream:
        # Create trace:
        tr_to_add = Trace()
        tr_to_add.stats.station = das_station_labels[i]
        tr_to_add.data = data[:, das_station_idxs[i]].astype(float)
        tr_to_add.stats.sampling_rate = fs
        tr_to_add.stats.starttime = starttime_curr
        tr_to_add.stats.channel = "EHN"
        tr_to_add.stats.network = network_code
        # Append trace to stream:
        st.append(tr_to_add)

        # Duplicate for Z and E components (arbitarily set equal to N comp),
        # if specified:
        if duplicate_Z_and_E:
            tr_to_add_Z = tr_to_add.copy()
            tr_to_add_Z.stats.channel = "EHZ"
            st.append(tr_to_add_Z)
            tr_to_add_E = tr_to_add.copy()
            tr_to_add_E.stats.channel = "EHE"
            st.append(tr_to_add_E)
            del tr_to_add_Z, tr_to_add_E
        
        # Tidy memory:
        del tr_to_add

    # Tidy memory:
    del data_and_headers, headers, data 

    return st 


def read_das_segy(das_fname, first_last_das_channels=[0,-1], network_code="AA", station_prefix="D", 
                spatial_down_samp_factor=1, fk_filter_params={}, duplicate_Z_and_E=True, 
                apply_notch_filter=False, notch_freqs=[], notch_bw=2.5, semblance_stack=False,
                semblance_v_app_min=1.0, channel_spacing=None, gauge_length=None, 
                convert_strainrate_to_vel=False):
    """Function to read in single das h5 file and output as obspy stream object."""
    # Check essential inputs are specified:
    if channel_spacing == None:
        raise util.DASSEGYNoChSpacSpecException
    if gauge_length == None:
        raise util.DASSEGYNoGLException

    # Get start time of data:
    st_in = read(das_fname, format="SEGY")
    das_start = st_in[0].stats.starttime

    # Create station labels:
    # (based on distance along fibre)
    end_channel = first_last_das_channels[1] 
    if end_channel == -1:
        end_channel = len(st_in)
    das_station_idxs = np.arange(first_last_das_channels[0], end_channel, int(spatial_down_samp_factor), dtype=int)
    das_station_labels = []
    for idx in das_station_idxs:
        das_station_labels.append(station_prefix+str(round(idx*channel_spacing)).zfill(4))

    # And process data:
    starttime_curr = das_start
    fs = st_in[0].stats.sampling_rate
    headers = {}
    headers['fs'] = fs
    headers['dx'] = channel_spacing
    headers['gauge'] = gauge_length

    # Create 2D das data object:
    data = np.zeros((len(st_in[0].data), len(st_in))) # (data of shape (time_samp, spatial_samp))
    for i in range(len(st_in)):
        data[:,i] = st_in[i].data
    del st_in

    # Filter data:
    # Apply fk filter:
    if len(list(fk_filter_params.keys())) > 0:
        print('Applying fk filter')
        data = fk_filter(data, fs, channel_spacing, fk_filter_params['wavenumber'], fk_filter_params['max_freq'], 
                        v_app_filts=fk_filter_params['v_app_filts'])
    # Apply notch filter:
    if apply_notch_filter:
        print('Applying notch filter/s')
        for f_notch in notch_freqs:
            data = notch_filter(data, fs, f_notch, notch_bw, filt_axis=0)

    # Convert data to velocity, if specified:
    if convert_strainrate_to_vel:
        print("Converting das strain-rate to velocity.")
        data = strainrate2vel(data, headers)
    
    # Perform semblance stack, if decimating data and semblance stacking is specified:
    if not spatial_down_samp_factor==1:
        if semblance_stack:
            win_len = int(0.5*fs) # Set window length to 1/2 a second
            max_inter_ch_t_shift = int(np.ceil(channel_spacing / (semblance_v_app_min * 1000))) # (This should be based 
            # on slowest apparent velocity, i.e. dx/v_app, in samples)
            data = semblance_stack_all(data, win_len, spatial_down_samp_factor, max_inter_ch_t_shift=max_inter_ch_t_shift)
    
    # Loop over das channels to save:
    st = Stream()
    for i in range(len(das_station_labels)):
        # Add data to stream:
        # Create trace:
        tr_to_add = Trace()
        tr_to_add.stats.station = das_station_labels[i]
        tr_to_add.data = data[:, das_station_idxs[i]].astype(float)
        tr_to_add.stats.sampling_rate = fs
        tr_to_add.stats.starttime = starttime_curr
        tr_to_add.stats.channel = "EHN"
        tr_to_add.stats.network = network_code
        # Append trace to stream:
        st.append(tr_to_add)

        # Duplicate for Z and E components (arbitarily set equal to N comp),
        # if specified:
        if duplicate_Z_and_E:
            tr_to_add_Z = tr_to_add.copy()
            tr_to_add_Z.stats.channel = "EHZ"
            st.append(tr_to_add_Z)
            tr_to_add_E = tr_to_add.copy()
            tr_to_add_E.stats.channel = "EHE"
            st.append(tr_to_add_E)
            del tr_to_add_Z, tr_to_add_E
        
        # Tidy memory:
        del tr_to_add

    # Tidy memory:
    del headers, data 

    return st 



def fk_filter(data, fs, ch_space, wavenumber, max_freq, v_app_filts=None, small_wavenumbers_filt_range=None, plot=False):
    """FK filter for a 2D DAS numpy array. Returns a filtered image.
    Originally created by Antony Butcher.
    data - 2D array to filter. Data must be of shape (time_samp, spatial_samp). (np array)
    fs - The sampling rate. (float)
    ch_space - Channel spacing, in metres. (float)
    wavenumber - Wavenumber for fk filter. (float)
    max_freq - Maximum frequency for fk filter, in Hz. (float)
    v_app_filts - If specified, will remove these specific apparent velocities (w/k). (list of floats)
    small_wavenumbers_filt_range - If specified, will remove wavenumbers between this range
    """
    # Detrend by removing the mean 
    data=data-np.mean(data)
    
    # Apply a 2D fft transform
    fftdata=np.fft.fftshift(np.fft.fft2(data.T))
    freqs=np.fft.fftfreq(fftdata.shape[1],d=(1./fs))
    wavenums=np.fft.fftfreq(fftdata.shape[0],d=ch_space)
    freqs=np.fft.fftshift(freqs) 
    wavenums=np.fft.fftshift(wavenums)
    freqsgrid=np.broadcast_to(freqs,fftdata.shape)   
    wavenumsgrid=np.broadcast_to(wavenums,fftdata.T.shape).T
    
    # Define mask and blur the edges 
    mask=np.logical_and(np.logical_and(wavenumsgrid<=wavenumber,wavenumsgrid>=-wavenumber),abs(freqsgrid)<max_freq)
    x=mask*1.
    blurred_mask = ndimage.gaussian_filter(x, sigma=3)
    
    # Apply the mask to the data
    ftimagep = fftdata * blurred_mask
    
    # Define and apply apparent velocity mask, if specifed:
    dk = 2*np.abs(wavenums[1] - wavenums[0])
    df = 2*np.abs(freqs[1] - freqs[0])
    if not v_app_filts==None:
        for app_v in v_app_filts:
            app_v_mask = np.logical_and(np.logical_or(wavenumsgrid<=2*np.pi*(abs(freqsgrid)-df)/app_v, wavenumsgrid>=2*np.pi*(abs(freqsgrid)+df)/app_v),
                                        np.logical_or(abs(freqsgrid)<app_v*(np.abs(wavenumsgrid)-dk)/(2*np.pi),abs(freqsgrid)>app_v*(np.abs(wavenumsgrid)+dk)/(2*np.pi)))
            x=app_v_mask*1.
            blurred_app_v_mask = ndimage.gaussian_filter(x, sigma=1)
            ftimagep = ftimagep * blurred_app_v_mask

    # Remove small wavenumbers, if specifed:
    if not small_wavenumbers_filt_range==None:
        small_k_mask = np.logical_or(wavenumsgrid<=small_wavenumbers_filt_range[0], wavenumsgrid>=small_wavenumbers_filt_range[1])
        x=small_k_mask*1.
        blurred_small_k_mask = ndimage.gaussian_filter(x, sigma=1)
        ftimagep = ftimagep * blurred_small_k_mask

    # Shift the ifft:
    ftimagep = np.fft.ifftshift(ftimagep)
    
    # Finally, take the inverse transform and show the blurred image
    imagep = np.fft.ifft2(ftimagep)
    imagep = imagep.real
    imagep = imagep.T

    return imagep
    

def notch_filter(data, fs, f_notch, bw, filt_axis=-1):
    """Notch filter to filter out a specific frequency.
    Note: Applies a zero phase filter.
    Argments:
    data - The time series to filter (np array)
    fs - The sampling rate, in Hz (float)
    f_notch - The frequency to apply a notch filter for in Hz (float)
    bw - The bandwidth of the notch filter
    filt_axis - The axis to apply the filter to

    Returns:
    data_filt - The filtered time series.
    """
    # Create the notch filter:
    Q = float(f_notch) / float(bw)
    b, a = signal.iirnotch(f_notch, Q, fs)
    # Apply filter (zero phase):
    data_filt = signal.filtfilt(b, a, data, axis=filt_axis)
    return data_filt


def semblance(data, max_inter_ch_t_shift=2):
    """Calculates semblance values for given window and shift.
    Shift is applied relative to centre channel. Data should be 
    of shape (time, space).
    Note: Shifts here are relative to first trace, not centre trace."""
    n_ch = data.shape[1]
    t_shifts = np.arange(-max_inter_ch_t_shift,max_inter_ch_t_shift+1, dtype=int)

    # Set initial values for semblance max. search:
    semb_max = 0.
    data_stacked = np.sum(data, axis=1) / n_ch
    # shifts_max = np.zeros(len(t_shifts), dtype=int)
    
    # Roll each channel in turn, finding max. semblance and keeping match:
    # (Note: Shifts here are relative to first trace, not centre trace)
    for ch in range(1,n_ch):
        for t_shift in t_shifts:
            # Time shift next channel:
            data[:,ch] = np.roll(data[:,ch], t_shift, axis=0)
            # Calculate current semblance:
            semb_curr = (1/n_ch) * np.sum(np.sum(data, axis=1)**2) / np.sum(np.sum(data**2, axis=1))
            # And update if semblance is a maximum:
            if semb_curr > semb_max:
                semb_max = semb_curr.copy()
                data_stacked = np.sum(data, axis=1) / n_ch
                # shifts_max[ch] = t_shift
    
    return data_stacked


def semblance_stack_all(data, win_len, ch_dec_fac, max_inter_ch_t_shift=2):
    """Function to perform semblance stacking on all das data, given some windows."""
    for ch_start in range(0, data.shape[0]-ch_dec_fac, ch_dec_fac):
        for win_start in range(0, data.shape[0]-win_len, win_len):
            data_stacked = semblance(data[win_start:win_start+win_len,ch_start:ch_start+ch_dec_fac], 
                                        max_inter_ch_t_shift=max_inter_ch_t_shift)
            # And append data, based on whether at end or not:
            fill_dim = data[win_start:win_start+win_len,ch_start:ch_start+ch_dec_fac].shape[1]
            data[win_start:win_start+win_len,ch_start:ch_start+ch_dec_fac] = np.repeat(data_stacked, 
                                                        fill_dim, axis=0).reshape(win_len, fill_dim)
    del data_stacked 
    return data


def _direct_integration(twoD_data_arr, GL=1, dx=1, axis=0):
    """Function to perform direct integration of <data_arr> along a particular axis. Note that detrends data, to remove drift.
    Note: Only takes 2D data."""
    # And perform integration
    twoD_data_arr_int = twoD_data_arr.copy()
    if axis == 0:
        for i in range(twoD_data_arr.shape[1]):
            y = twoD_data_arr[:,i]
            y = y - np.mean(y) # detrend data
            y_int = integrate.cumtrapz(y, dx=dx)
            y_int = np.append(y_int, y_int[-1]) # (and set final value, as not calculated otherwise)
            twoD_data_arr_int[:,i] = y_int - np.mean(y_int) # And detrend data
    else:
        for i in range(twoD_data_arr.shape[0]):
            y = twoD_data_arr[i,:]
            y = y - np.mean(y) # detrend data
            y_int = integrate.cumtrapz(y, dx=dx)
            y_int = np.append(y_int, y_int[-1]) # (and set final value, as not calculated otherwise)
            twoD_data_arr_int[i,:] = y_int - np.mean(y_int) # And detrend data
    return twoD_data_arr_int


def strainrate2vel(data, headers, fk_filter_params=None, 
                    bp_filter_params=None, notch_filter_params=None, verbosity=0):
    """
    Function to calculate strain rate from data.

    Parameters
    ----------
    data, headers : 
        H5 data and headers, in ETH SWP format.
    
    first_ch : int
        The first channel to load. Default = 0.

    last_ch : int 
        The last channel to load. Default = None, which loads all channels.

    first_s : int
        The first time sample to load. Defualt = 0.

    last_s : int
        The last time sample to load. Default = None, which loads all 
        channels.

    fk_filter_params : dict
        Dictionary containing fk filter parameters. Will only apply 
        fk-filter if specified. Example format of dictionary is:
        fk_filter_params['wavenumber'] = 0.04
        fk_filter_params['max_freq'] = 100. (In Hz)
        Default is None, i.e. no filter applied.

    bp_filter_params : dict
        Dictionary containing bandpass filter parameters. Will only apply 
        bandpass filter if specified. Example format of dictionary is:
        bp_filter_params['filter_freqs'] = [1.0, 150.0] (in Hz)
        Default is None, i.e. no filter applied.

    notch_filter_params : dict
        Dictionary containing notch filter parameters. Will only apply 
        notch filter if specified. Example format of dictionary is:
        notch_filter_params['notch_freqs'] = [33.0, 66.0] (in Hz)
        notch_filter_params['notch_bw'] = 2.5 (in Hz)
        Default is None, i.e. no filter applied.

    Returns
    -------
    vel_data : np array
        Array containing DAS data converted to velocity from <tdms> data 
        input. Shape is (time_samples, das_channels), as in original tdms 
        data format.

    """
    # 1. Get data and properties:
    strain_rate_data = data - np.mean(data) #data[first_s:last_s, first_ch:last_ch] - np.mean(data[first_s:last_s, first_ch:last_ch]) # Demean data
    fs = headers['fs']
    dx = headers['dx']
    GL = headers['gauge']

    # 2. Filter data:
    # fk filter params:
    if fk_filter_params:
        strain_rate_data = fk_filter(strain_rate_data, fs, dx, fk_filter_params['wavenumber'], 
                                        fk_filter_params['max_freq'], plot=False)
    if bp_filter_params:
        strain_rate_data = twoD_bandpass_filter(strain_rate_data, bp_filter_params['filter_freqs'][0], 
                                        bp_filter_params['filter_freqs'][1], fs, order=4, axis=0)
    if notch_filter_params:
        for f_notch in notch_filter_params['notch_freqs']:
            strain_rate_data = notch_filter(strain_rate_data, fs, f_notch, notch_filter_params['notch_bw'], 
                                            filt_axis=0)

    # 3. Convert strain-rate to velocity:
    # via direct integration method:
    # (Integrate data spatially):
    vel_data = _direct_integration(strain_rate_data, GL=GL, dx=dx, axis=1)
    vel_data = vel_data / GL # To correct for gauge length effect (Don't need to apply as integrating over each 
    #                           spatial sample rather than each gauge length (?))

    # 4. And perform fk filter to remove infinite apparent velocity spatial integration noise:
    max_wavenum = 1. / dx
    max_freq = fs / 2.
    fibre_len = float(strain_rate_data.shape[1]) * dx
    small_wavenumbers_filt_range = [-4./fibre_len, 4./fibre_len] # (4 is an empirically derived factor, determining 
    #                                                           width of zero app freq. filter. Could be less or more, 
    #                                                           if datasets vary).
    vel_data = fk_filter(vel_data, fs, dx, max_wavenum, max_freq, small_wavenumbers_filt_range=small_wavenumbers_filt_range)

    return vel_data
