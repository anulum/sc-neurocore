# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike waveform shape analysis

"""Spike waveform shape analysis."""

from __future__ import annotations

from typing import Any
import numpy as np


def waveform_width(waveform: np.ndarray[Any, Any], dt: float = 1.0 / 30000) -> float:
    """Trough-to-peak width (seconds). Bartho et al. 2004.

    Measures time from waveform minimum to subsequent maximum.
    """
    trough = np.argmin(waveform)
    if trough >= waveform.size - 1:
        return float("nan")
    peak = trough + np.argmax(waveform[trough:])
    return float((peak - trough) * dt)


def waveform_amplitude(waveform: np.ndarray[Any, Any]) -> float:
    """Peak-to-trough amplitude. Bartho et al. 2004."""
    return float(np.max(waveform) - np.min(waveform))


def waveform_repolarization_slope(waveform: np.ndarray[Any, Any], dt: float = 1.0 / 30000) -> float:
    """Repolarization slope: max dV/dt after trough. Bean 2007."""
    trough = np.argmin(waveform)
    if trough >= waveform.size - 2:
        return float("nan")
    post_trough = waveform[trough:]
    dv = np.diff(post_trough) / dt
    return float(np.max(dv))


def waveform_recovery_slope(waveform: np.ndarray[Any, Any], dt: float = 1.0 / 30000) -> float:
    """Recovery slope: dV/dt during return to baseline after peak. Bean 2007."""
    trough = np.argmin(waveform)
    if trough >= waveform.size - 1:
        return float("nan")
    peak = trough + np.argmax(waveform[trough:])
    if peak >= waveform.size - 2:
        return float("nan")
    # The guard above ensures peak <= size - 3, so post_peak spans at least
    # three samples and np.diff yields at least two -- dv is never empty here.
    post_peak = waveform[peak:]
    dv = np.diff(post_peak) / dt
    return float(np.min(dv))


def waveform_halfwidth(waveform: np.ndarray[Any, Any], dt: float = 1.0 / 30000) -> float:
    """Half-width: duration at half-minimum amplitude. Bartho et al. 2004."""
    trough_val = np.min(waveform)
    half_val = trough_val / 2.0
    below = np.where(waveform < half_val)[0]
    if below.size < 2:
        return float("nan")
    return float((below[-1] - below[0]) * dt)


def waveform_pt_ratio(waveform: np.ndarray[Any, Any]) -> float:
    """Peak-to-trough ratio. Bartho et al. 2004.

    Ratio of post-trough peak amplitude to trough amplitude.
    """
    trough = np.argmin(waveform)
    trough_val = abs(waveform[trough])
    if trough >= waveform.size - 1 or trough_val < 1e-30:
        return float("nan")
    peak_val = np.max(waveform[trough:])
    return float(abs(peak_val) / trough_val)
