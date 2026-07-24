# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_sources_and_profiling.py

from __future__ import annotations

"""
Tests for untested source and profiling modules:
  - sources/quantum_entropy.py (QuantumEntropySource)
  - sources/bitstream_current_source.py (BitstreamCurrentSource)
  - profiling/energy.py (EnergyMetrics, track_energy)
"""
import pytest
import numpy as np
from sc_neurocore.sources.quantum_entropy import QuantumEntropySource
from sc_neurocore.sources.bitstream_current_source import BitstreamCurrentSource
from sc_neurocore.profiling.energy import EnergyMetrics, profiler, track_energy

__all__ = [
    "pytest",
    "np",
    "QuantumEntropySource",
    "BitstreamCurrentSource",
    "EnergyMetrics",
    "profiler",
    "track_energy",
]
