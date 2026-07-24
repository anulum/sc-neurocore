# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_spade_gpfa.py

from __future__ import annotations

"""Tests for SPADE and GPFA modules."""
import numpy as np
from sc_neurocore.analysis.spike_stats import spade_detect, gpfa, gpfa_transform
from sc_neurocore.analysis.spike_stats.spade import (
    _find_frequent_itemsets,
)
from sc_neurocore.analysis.spike_stats.gpfa import _gp_kernel, _gpfa_log_likelihood


def _sync_trains(n_neurons=5, n_steps=2000, sync_every=100, seed=42):
    """Generate spike trains with a planted synchronous pattern."""
    rng = np.random.default_rng(seed)
    trains = []
    for i in range(n_neurons):
        t = np.zeros(n_steps, dtype=np.uint8)
        spikes = rng.choice(n_steps, size=20, replace=False)
        t[spikes] = 1
        trains.append(t)
    # Plant synchronous events for neurons 0,1,2 at regular intervals
    for offset in range(0, n_steps, sync_every):
        if offset < n_steps:
            for nid in [0, 1, 2]:
                trains[nid][offset] = 1
    return trains


def _poisson_trains(n_neurons=6, rate_hz=30.0, duration_s=1.0, dt=0.001, seed=42):
    rng = np.random.default_rng(seed)
    n_steps = int(duration_s / dt)
    trains = []
    for _ in range(n_neurons):
        t = (rng.random(n_steps) < rate_hz * dt).astype(np.uint8)
        trains.append(t)
    return trains


__all__ = [
    "np",
    "spade_detect",
    "gpfa",
    "gpfa_transform",
    "_find_frequent_itemsets",
    "_gp_kernel",
    "_gpfa_log_likelihood",
    "_sync_trains",
    "_poisson_trains",
]
