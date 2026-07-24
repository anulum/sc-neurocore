# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_waveform_codec.py

from __future__ import annotations

import struct
import sys
from typing import Any
import numpy as np
import pytest
from sc_neurocore.spike_codec.waveform_codec import WaveformCodec


def _make_waveform(
    T: int = 2000,
    N: int = 16,
    noise_sigma: float = 50.0,
    spike_rate: float = 3.0,
    seed: int = 42,
) -> np.ndarray[Any, Any]:
    """Generate synthetic raw electrode waveform with spikes."""
    rng = np.random.RandomState(seed)
    waveform = rng.randn(T, N).astype(np.float32) * noise_sigma
    template = np.zeros(48)
    template[15:20] = -200
    template[20:25] = 100
    template[25:30] = -30
    for ch in range(N):
        n_spikes = max(1, int(spike_rate * T / 20000))
        times = rng.choice(range(100, T - 100), size=min(n_spikes, T - 200), replace=False)
        for t in times:
            s, e = max(0, t - 24), min(T, t + 24)
            waveform[s:e, ch] += template[: e - s]
    return waveform


__all__ = ["struct", "sys", "Any", "np", "pytest", "WaveformCodec", "_make_waveform"]
