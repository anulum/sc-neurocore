# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_spade_gpfa.py

"""Shared spike-train generators for the SPADE and GPFA suites."""

from __future__ import annotations

from typing import Any

import numpy as np


def _sync_trains(
    n_neurons: int = 5, n_steps: int = 2000, sync_every: int = 100, seed: int = 42
) -> list[np.ndarray[Any, Any]]:
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


def _poisson_trains(
    n_neurons: int = 6,
    rate_hz: float = 30.0,
    duration_s: float = 1.0,
    dt: float = 0.001,
    seed: int = 42,
) -> list[np.ndarray[Any, Any]]:
    """Generate independent Poisson spike trains at a fixed rate."""
    rng = np.random.default_rng(seed)
    n_steps = int(duration_s / dt)
    trains = []
    for _ in range(n_neurons):
        t = (rng.random(n_steps) < rate_hz * dt).astype(np.uint8)
        trains.append(t)
    return trains


__all__ = [
    "_sync_trains",
    "_poisson_trains",
]
