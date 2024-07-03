# -*- coding: utf-8 -*-
"""
Module to handle input/output of cut waveforms.

:copyright:
    2020–2024, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import logging
import warnings
import numpy as np 
import quakemigrate.util as util


def write_coa_map(
    run,
    event,
    fourD_output=False
):
    """
    Output 3D (or 4D) coalescence map at event origin time.

    Parameters
    ----------
    run : :class:`~quakemigrate.io.core.Run` object
        Light class encapsulating i/o path information for a given run.
    event : :class:`~quakemigrate.io.event.Event` object
        Light class encapsulating waveforms, coalescence information, picks and location
        information for a given event.
    fourD_output : bool 
        If True, outputs 4D output rather than 3D. Defaults to False. Note that 4D grids 
        can use a lot of storage space.

    Raises
    ------

    """

    logging.info(f"\tSaving coa map...")

    # Create paths to save files to:
    fpath = run.path / "locate" / run.subname / f"coa_maps"
    fpath.mkdir(exist_ok=True, parents=True)
    fstem = f"{event.uid}"

    # Write coalescence map to file:
    try:
        file = (fpath / fstem).with_suffix(".npy")
        if fourD_output:
            np.save(str(file), event.map4d)
        else:
            np.save(str(file), np.sum(event.map4d, axis=-1))
    except:
        logging.info(f"\t\tNo coalescence map data for event{event.uid}!")


