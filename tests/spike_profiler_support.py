# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_spike_profiler.py

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.profiling.spike_profiler import (
    SpikeProfiler,
    LayerStats,
    ProfileReport,
    Pathology,
    Severity,
)
def _random_spikes(n_neurons, rate=0.1, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    return (rng.random(n_neurons) < rate).astype(np.int8)

__all__ = ['np', 'pytest', 'SpikeProfiler', 'LayerStats', 'ProfileReport', 'Pathology', 'Severity', '_random_spikes']
