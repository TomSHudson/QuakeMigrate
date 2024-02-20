# -*- coding: utf-8 -*-
"""
The :mod:`quakemigrate.io` module handles the various input/output operations performed
by QuakeMigrate. This includes:

    * Reading waveform data - The submodule data.py can handle any waveform \
      data archive with a regular directory structure. It also provides \
      functions for checking data quality and removing/simulating instrument \
      reponse.
    * Reading station files, velocity model files, instrument response \
      inventories and QuakeMigrate lookup tables.
    * The :class:`~quakemigrate.io.core.Run` class encapsulates all i/o path \
      information and logger configuration for a given QuakeMigrate run.
    * The :class:`~quakemigrate.io.event.Event` class encapsulates waveforms, \
      coalescence information, picks and location information for a given \
      event, and provides functionality to write ".event" files.
    * Reading and writing results, including station availablity data and \
      continuous coalescence output from detect; triggered event files from \
      trigger, amplitude and local magnitude measurements and cut waveforms \
      for located events.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from .das import read_das  # NOQA