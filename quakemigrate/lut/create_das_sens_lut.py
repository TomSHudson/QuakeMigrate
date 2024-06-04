# -*- coding: utf-8 -*-
"""
Module to produce traveltime lookup tables defined on a Cartesian grid.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import logging
import warnings
import os, sys
import pathlib
import struct
from shutil import rmtree
import gc 

import numpy as np
from pyproj import Proj, Transformer
from scipy.interpolate import interp1d

import ttcrpy.rgrid as ttcrpy_rgrid

import quakemigrate.util as util
from .lut import LUT


def _find_nearest_val(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def compute_das_sensitivity(
    lut,
    grid_spec,
    station_prefix="D",
    strainrate_or_velocity="velocity",
    das_sens_lower_cutoff=0.1,
    write_out_das_sensitivities=False,
    save_file=None,
    log=False,
):
    """
    Top-level method for computing DAS sensitivity lookup tables.

    This function takes a grid specification and is capable of computing DAS sensitivities
    for an P and/or S phases.

    Parameters
    ----------
    lut : :class:`~quakemigrate.lut.lut.LUT` object
        Lookup table populated with traveltimes.
    grid_spec : dict
        Dictionary containing all of the defining parameters for the underlying 3-D grid
        on which the traveltimes are to be calculated. For expected keys, see
        :class:`~quakemigrate.lut.lut.Grid3D`.
    station_prefix : str, optional
        Station name prefix for das channels. das channels are then named with the 
        station prefix followed by an integer representing the distance along the fibre.
        Default is "D".
    strainrate_or_velocity : str, optional
        Specifies whether output sensitivity is for strain-rate or velocity. Default is 
        velocity. One should be consistent with using the correct sensitivity for the 
        correct measurement/conversion specified in detect/locate stages.
    das_sens_lower_cutoff : float, optional
        The value below which any lut nodes are deemed to be insensitive to P/S phases. 
        Default is 0.1, where sensitivity ranges from 0 to 1.
    write_out_das_sensitivities : bool, optional
        If True, writes out das_sensitivities to lut.das_sensitivities. Default is False.
    save_file : str, optional
        Path to location to save pickled lookup table.
    log : bool, optional
        Toggle for logging - default is to only print information to stdout.
        If True, will also create a log file.

    Returns
    -------
    lut : :class:`~quakemigrate.lut.lut.LUT` object
        Lookup table populated with traveltimes and DAS sensitivity lookup tables.

    Raises
    ------

    """

    util.logger(pathlib.Path.cwd() / "logs" / "lut", log)

    # 1. Specify velocity grid:
    ztmp = np.linspace(lut.grid_extent[0,2], lut.grid_extent[1,2]+lut.cell_size[2], lut.cell_count[2])
    # Set velocity based on homogeneious or 1D (3D not yet supported):
    try:
        # If 1D:
        vmodel_oneD_P = np.interp(ztmp, lut.velocity_model['Depth'], lut.velocity_model['Vp'])
        vmodel_oneD_S = np.interp(ztmp, lut.velocity_model['Depth'], lut.velocity_model['Vs'])
    except TypeError:
        # If homogeneous:
        vp_tmp = float(lut.velocity_model.split('=  ')[1].split(' m')[0])
        vs_tmp = float(lut.velocity_model.split('=  ')[2].split(' m')[0])
        vmodel_oneD_P = vp_tmp * np.ones(lut.grid_xyz[0].shape[2])
        vmodel_oneD_S = vs_tmp * np.ones(lut.grid_xyz[0].shape[2])
    # And populate 3D velocity grid accordingly:
    velocity_grid_P = np.zeros(lut.grid_xyz[0].shape)
    velocity_grid_S = np.zeros(lut.grid_xyz[0].shape)
    for i in range(lut.grid_xyz[0].shape[0]):
        for j in range(lut.grid_xyz[0].shape[1]):
            velocity_grid_P[i,j,:] = vmodel_oneD_P
            velocity_grid_S[i,j,:] = vmodel_oneD_S

    # 2. Calc. fibre directions from channel to channel:
    rec_vecs = np.zeros(lut.stations_xyz.shape)
    rec_vecs[:-1,0] = lut.stations_xyz[1:,0] - lut.stations_xyz[:-1,0]
    rec_vecs[:-1,1] = lut.stations_xyz[1:,1] - lut.stations_xyz[:-1,1]
    rec_vecs[:-1,2] = lut.stations_xyz[1:,2] - lut.stations_xyz[:-1,2]

    # 3. Calc. sensitivity for all LUT grid locs:
    print("Warning: Calculating DAS sensitivity only for 2D (horizontal) fibre geometry for now...")
    lut.das_sensitivities = {}
    # Loop over receivers:
    print("Computing DAS sensitivity for...")
    for i in range(lut.stations_xyz.shape[0]):
        print("...station: "+lut.station_data.loc[i].Name+" - "+str(i+1)+" of "+str(lut.stations_xyz.shape[0]))

        # Check whether channel is a DAS channel:
        if lut.station_data.loc[i].Name[0] == station_prefix:
            if len(lut.station_data.loc[i].Name[1:]) == 4:
                is_das = True
            else:
                is_das = False
        else:
            is_das = False

        # If a DAS channel:
        if is_das:
            # Get toas:
            station_xyz = lut.stations_xyz[i,:]
            toas_P = _find_toa_grid_single_receiver(np.array(lut.grid_xyz), lut.cell_size, velocity_grid_P, station_xyz)
            toas_S = _find_toa_grid_single_receiver(np.array(lut.grid_xyz), lut.cell_size, velocity_grid_S, station_xyz)
            
            # And find sensitivities:
            # Define angles:
            fibre_vec = rec_vecs[i,:] # [x, y] (Note: Currently only 2D, as assume in horizontal plane for now)
            # Fibre angle (from x):
            theta = np.arctan2(fibre_vec[1], fibre_vec[0])
            # Angles of incoming waves (from x):
            phi1s = np.arctan2(lut.grid_xyz[1].flatten()-station_xyz[1], lut.grid_xyz[0].flatten()-station_xyz[0]) # Azimuth
            phi2s_P = (np.pi/2.) - np.deg2rad(toas_P.flatten()) # (pi/2 - toa as phi2 is from horizontal (???)) $ TOA
            phi2s_S = (np.pi/2.) - np.deg2rad(toas_S.flatten()) # (pi/2 - toa as phi2 is from horizontal (???)) $ TOA

            # And calculate sensitivity (with no frequency dependence or gauge-length effects):
            if strainrate_or_velocity == "velocity":
                P_sens_vel = np.abs(np.cos(phi1s - theta) * np.cos(phi2s_P))
                SV_sens_vel = np.abs(np.cos(phi1s - theta) * np.sin(phi2s_S))
                SH_sens_vel = np.abs(np.sin(phi1s - theta))
            else:
                P_sens_vel = (np.cos(phi1s - theta) * np.cos(phi2s_P))**2
                SV_sens_vel = np.abs((np.cos(phi1s - theta)**2) * np.sin(2*phi2s_S))
                SH_sens_vel = np.abs(np.sin(2*(phi1s - theta)) * np.cos(phi2s_S))
            P_sens_vel = P_sens_vel.reshape(lut.grid_xyz[0].shape)
            SV_sens_vel = SV_sens_vel.reshape(lut.grid_xyz[0].shape)
            SH_sens_vel = SH_sens_vel.reshape(lut.grid_xyz[0].shape)
            # Combine SV and SH:
            S_sens_vel = np.sqrt((SV_sens_vel**2 + SH_sens_vel**2) / 2.) # (Note: Currently calculates S-wave sensitivity 
            # as RMS of SV and SH)
            S_sens_vel[SV_sens_vel>S_sens_vel] = SV_sens_vel[SV_sens_vel>S_sens_vel]
            S_sens_vel[SH_sens_vel>S_sens_vel] = SH_sens_vel[SH_sens_vel>S_sens_vel]

            # And set traveltimes to zero if DAS sensitivity below a given threshold:
            lut.traveltimes[lut.station_data.loc[i].Name]['P'][P_sens_vel<das_sens_lower_cutoff] = 0.
            lut.traveltimes[lut.station_data.loc[i].Name]['S'][S_sens_vel<das_sens_lower_cutoff] = 0.
            
            # And append to data stores:
            if write_out_das_sensitivities:
                lut.das_sensitivities[lut.station_data.loc[i].Name] = {}
                lut.das_sensitivities[lut.station_data.loc[i].Name]['P'] = P_sens_vel
                lut.das_sensitivities[lut.station_data.loc[i].Name]['S'] = S_sens_vel

        # Or if isn't a DAS channel, set sensitivity to 1 everywhere:
        else:
            if write_out_das_sensitivities:
                lut.das_sensitivities[lut.station_data.loc[i].Name] = {}
                lut.das_sensitivities[lut.station_data.loc[i].Name]['P'] = np.ones(lut.traveltimes[lut.station_data.loc[i].Name]['P'].shape)
                lut.das_sensitivities[lut.station_data.loc[i].Name]['S'] = np.ones(lut.traveltimes[lut.station_data.loc[i].Name]['P'].shape)

    # And tidy:
    try:
        del P_sens_vel, S_sens_vel, toas_P, toas_S, velocity_grid_P, velocity_grid_S, vmodel_oneD_P, vmodel_oneD_S, ztmp
    except UnboundLocalError:
        del velocity_grid_P, velocity_grid_S, vmodel_oneD_P, vmodel_oneD_S, ztmp
    gc.collect()

    

    if save_file is not None:
        lut.save(save_file)

    return lut


def _find_toa_grid_single_receiver(grid_xyz, node_spacing, velocity_grid, station_xyz):
    """
    Calculates the takeoff angles for a single receiver for sources from every grid point.

    .. warning:: Requires the ttcrpy (and vtk) python packages.

    Parameters
    ----------
    grid_xyz : array-like
        [X, Y, Z] coordinates of each node.
    node_spacing : array-like
        [X, Y, Z] distances between each node.
    velocity_grid : array-like
        Contains the speed of interface propagation at each point in the domain.
    station_xyz : array-like
        Station location (in grid xyz).

    Returns
    -------
    toas : array-like, same shape as grid_xyz
        Contains the takeoff angle from each point in grid_xyz.

    Raises
    ------
    ImportError
        If ttcrpy is not installed.

    """

    try:
        import ttcrpy
    except ImportError:
        raise ImportError(
            "Unable to import ttcrpy - you need to install ttcrpy (and vtk) to use this "
            "method."
        )

    # Pad grid, to remove any ray-tracing out of grid issues:
    # (Note: Important that don't update grid_xyz, as that is used for defining sources/receivers)
    # Pad in x, y and z:
    n_cells_pad = 5
    xyz = [grid_xyz[0][:,0,0], grid_xyz[1][0,:,0], grid_xyz[2][0,0,:]]
    xyz_with_pad = []
    # Loop over xyz:
    for xyz_idx in range(3):
        # Get pad labels:
        x_prepad = np.min(xyz[xyz_idx]) - np.arange(node_spacing[xyz_idx],(n_cells_pad+1)*node_spacing[xyz_idx], node_spacing[xyz_idx])[::-1]
        x_postpad = np.max(xyz[xyz_idx]) + np.arange(node_spacing[xyz_idx],(n_cells_pad+1)*node_spacing[xyz_idx], node_spacing[xyz_idx])
        xyz_with_pad.append(np.concatenate((x_prepad, xyz[xyz_idx], x_postpad)))
        # Pad velocity grid:
        if xyz_idx == 0:
            velocity_grid_prepad = velocity_grid[0,:,:].copy().reshape((1, velocity_grid.shape[1], velocity_grid.shape[2]))
            velocity_grid_postpad = velocity_grid[-1,:,:].copy().reshape((1, velocity_grid.shape[1], velocity_grid.shape[2]))
        elif xyz_idx == 1:
            velocity_grid_prepad = velocity_grid[:,0,:].copy().reshape((velocity_grid.shape[0], 1, velocity_grid.shape[2]))
            velocity_grid_postpad = velocity_grid[:,-1,:].copy().reshape((velocity_grid.shape[0], 1, velocity_grid.shape[2]))
        elif xyz_idx == 2:
            velocity_grid_prepad = velocity_grid[:,:,0].copy().reshape((velocity_grid.shape[0], velocity_grid.shape[1], 1))
            velocity_grid_postpad = velocity_grid[:,:,-1].copy().reshape((velocity_grid.shape[0], velocity_grid.shape[1], 1))
        for i in range(n_cells_pad):
            if xyz_idx == 0:
                velocity_grid = np.vstack((velocity_grid_prepad, velocity_grid, velocity_grid_postpad))
            elif xyz_idx == 1:
                velocity_grid = np.hstack((velocity_grid_prepad, velocity_grid, velocity_grid_postpad))
            elif xyz_idx == 2:
                velocity_grid = np.dstack((velocity_grid_prepad, velocity_grid, velocity_grid_postpad))

    # Create grid for ray-tracing:
    x = xyz_with_pad[0]
    y = xyz_with_pad[1]
    z = xyz_with_pad[2]
    try:
        rtgrid = ttcrpy_rgrid.Grid3d(x, y, z, cell_slowness=False)
    except ValueError as e:
        if str(e) != "FSM: Grid cells must be cubic":
            raise
        else:
            print("Error: Currently, lut grids must be cubic (i.e. all values of grid_spec.node_spacing must be equal).")
            sys.exit()
    
    # Define "source" and "receivers":
    src = station_xyz.reshape((1,3))
    nrcv = grid_xyz.shape[1] * grid_xyz.shape[2] * grid_xyz.shape[3]
    rcv = np.zeros((nrcv,3))
    rcv[:,0] = grid_xyz[0].flatten()
    rcv[:,1] = grid_xyz[1].flatten()
    rcv[:,2] = grid_xyz[2].flatten()

    # Perform ray tracing:
    tt, rays = rtgrid.raytrace(src, rcv, 1./velocity_grid, return_rays=True)
    del tt, rtgrid, rcv
    gc.collect()
    
    # And calculate takeoff angles for all points in grid:
    toas = np.zeros(len(rays))
    for i in range(len(rays)):
        # Deal with ray where source is at receiver:
        if rays[i].shape[0] < 4:
            toas[i] = 0.
        else:
            dxyz = rays[i][-4,:] - rays[i][-1,:]
#             dxyz = rays[i][-3,:] - rays[i][-2,:] # (Note: ignores point closest to receiver as spurious effect of ray-tracing here)
            dxy = np.sqrt(dxyz[0]**2 + dxyz[1]**2)
            dz = dxyz[2]
            toas[i] = np.rad2deg(np.arctan2(dxy, dz)) # Angle of ray at receiver, from vertical down
    toas = toas.reshape(grid_xyz[0].shape)
    del rays
    gc.collect()
    
    return toas


