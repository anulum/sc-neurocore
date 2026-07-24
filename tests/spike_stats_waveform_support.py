# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_spike_stats_waveform.py

from __future__ import annotations

"""Edge-case tests for every branch in waveform shape analysis functions."""
import numpy as np
from sc_neurocore.analysis.spike_stats.waveform import (
    waveform_width,
    waveform_amplitude,
    waveform_repolarization_slope,
    waveform_recovery_slope,
    waveform_halfwidth,
    waveform_pt_ratio,
)


def _typical_waveform():
    """Synthetic spike waveform: depolarisation → trough → repolarisation → overshoot → baseline."""
    # Realistic extracellular spike: brief negative trough then positive peak
    t = np.linspace(0, 2, 60)
    w = -np.exp(-((t - 0.4) ** 2) / 0.02) + 0.6 * np.exp(-((t - 0.8) ** 2) / 0.04)
    return w


__all__ = [
    "np",
    "waveform_width",
    "waveform_amplitude",
    "waveform_repolarization_slope",
    "waveform_recovery_slope",
    "waveform_halfwidth",
    "waveform_pt_ratio",
    "_typical_waveform",
]
