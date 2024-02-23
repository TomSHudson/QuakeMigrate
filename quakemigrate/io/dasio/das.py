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
from scipy import ndimage, signal
from obspy import read, Stream, Trace, UTCDateTime
import quakemigrate.util as util
import quakemigrate.io.dasio.load_das_h5 as load_das_h5


def read_das(das_archive_path, das_data_fmt, starttime, endtime, pre_pad=0.0, post_pad=0.0, 
            first_last_das_channels=[0,-1], duplicate_das_comps=True, station_prefix="D", 
            spatial_down_samp_factor=1, fk_filter_params={}, apply_notch_filter=False, 
            notch_freqs=[], notch_bw=2.5, semblance_stack=False, semblance_v_app_min=1.0):
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
        the das format to be read. Currently, the only supported format is h5, but it 
        is relatively trivial to support other formats in the future (contact the 
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
        st += read_das_h5(das_fname, first_last_das_channels=first_last_das_channels, station_prefix="D", 
                        spatial_down_samp_factor=spatial_down_samp_factor, fk_filter_params=fk_filter_params, 
                        duplicate_Z_and_E=duplicate_das_comps, apply_notch_filter=apply_notch_filter, 
                        notch_freqs=notch_freqs, notch_bw=notch_bw, semblance_stack=semblance_stack, 
                        semblance_v_app_min=semblance_stack)
    st = util.merge_stream(st)

    return st


def _get_das_starttime_from_fname(das_fname, das_data_fmt):
    """Function to get das starttime as UTCDateTime object from path object."""
    das_fname_tmp = str(pathlib.PurePath(das_fname).parts[-1])
    str_tmp = das_fname_tmp.split("."+das_data_fmt)[0]
    das_f_starttime_str = str_tmp.split("UTC_")[-1]
    das_f_starttime = UTCDateTime(year=int(das_f_starttime_str[0:4]), 
                                    month=int(das_f_starttime_str[4:6]),
                                    day=int(das_f_starttime_str[6:8]),
                                    hour=int(das_f_starttime_str[9:11]),
                                    minute=int(das_f_starttime_str[11:13]),
                                    second=int(das_f_starttime_str[13:15]),
                                    microsecond=int((10**6) * (10**(-1 * len(das_f_starttime_str[16:]))) 
                                                * int(das_f_starttime_str[16:])))
    return das_f_starttime



def read_das_h5(das_fname, first_last_das_channels=[0,-1], network_code="AA", station_prefix="D", 
                spatial_down_samp_factor=1, fk_filter_params={}, duplicate_Z_and_E=True, 
                apply_notch_filter=False, notch_freqs=[], notch_bw=2.5, semblance_stack=False,
                semblance_v_app_min=1.0):
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


def fk_filter(data, fs, ch_space, wavenumber, max_freq, v_app_filts=None, plot=False):
    """FK filter for a 2D DAS numpy array. Returns a filtered image.
    Originally created by Antony Butcher.
    data - 2D array to filter. Data must be of shape (time_samp, spatial_samp). (np array)
    fs - The sampling rate. (float)
    ch_space - Channel spacing, in metres. (float)
    wavenumber - Wavenumber for fk filter. (float)
    max_freq - Maximum frequency for fk filter, in Hz. (float)
    v_app_filts - If specified, will remove these specific apparent velocities (w/k). (list of floats)
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

    # Shift the ifft:
    ftimagep = np.fft.ifftshift(ftimagep)
    
    # Finally, take the inverse transform and show the blurred image
    imagep = np.fft.ifft2(ftimagep)
    imagep = imagep.real
    imagep = imagep.T
    
    # Plot the filter, if specified:
    if plot==True:
        # Plots the filter, with area remove greyed out
        fig, ax = plt.subplots(nrows=3, sharex=True, figsize=[4,8])
        img1 = ax[0].imshow(abs(fftdata), interpolation='bilinear',extent=[-fs/2,fs/2,-1/(2*ch_space),1/(2*ch_space)],aspect='auto')
        # img1.set_clim(-5,5)
        img1 = ax[0].imshow(abs(blurred_mask-1),cmap='Greys',extent=[-fs/2,fs/2,-1/(2*ch_space),1/(2*ch_space)],alpha=0.2,aspect='auto')
        ax[0].set_xlabel('Frequency (Hz)')
        ax[0].set_ylabel('Wavenumber (1/m)')
        ax[0].set_title("fk-filter, fk domain")

        # Plots second, app velocity filter, if specified:
        img1 = ax[1].imshow(abs(fftdata), interpolation='bilinear',extent=[-fs/2,fs/2,-1/(2*ch_space),1/(2*ch_space)],aspect='auto')
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Wavenumber (1/m)')
        ax[1].set_title("$v_{app}$ filter, fk domain")
        if not v_app_filts==None:
            img1 = ax[1].imshow(abs(blurred_app_v_mask-1),cmap='Greys',extent=[-fs/2,fs/2,-1/(2*ch_space),1/(2*ch_space)],alpha=0.2,aspect='auto')

        # Plots final removed data:
        img1 = ax[2].imshow(abs(np.fft.fftshift(ftimagep)), interpolation='bilinear',extent=[-fs/2,fs/2,-1/(2*ch_space),1/(2*ch_space)],aspect='auto')
        ax[2].set_xlabel('Frequency (Hz)')
        ax[2].set_ylabel('Wavenumber (1/m)')
        ax[2].set_title("Final data, fk domain")
        # plt.xlim(-200,200)
        # plt.ylim(-0.2,0.2)
        plt.show()

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
