# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-LFP coupling: phase locking, coherence, phase

"""Spike-LFP coupling: phase locking, coherence, phase histograms."""

from __future__ import annotations

from typing import Any
import numpy as np


def phase_locking_value(
    binary_train: np.ndarray[Any, Any], lfp_signal: np.ndarray[Any, Any]
) -> float:
    """Phase locking value (PLV) between spikes and LFP phase.

    Extracts instantaneous phase of LFP via Hilbert transform,
    then computes PLV = |mean(exp(j*phase_at_spikes))|.
    """
    n = min(binary_train.size, lfp_signal.size)
    analytic = np.fft.ifft(
        np.fft.fft(lfp_signal[:n].astype(np.float64)) * 2 * (np.arange(n) > 0).astype(np.float64)
    )
    phase = np.angle(analytic)
    spike_idx = np.where(binary_train[:n] > 0)[0]
    if spike_idx.size == 0:
        return 0.0
    return float(np.abs(np.mean(np.exp(1j * phase[spike_idx]))))


def spike_field_coherence(
    binary_train: np.ndarray[Any, Any], lfp_signal: np.ndarray[Any, Any], dt: float = 0.001
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Spike-field coherence (SFC) between binary train and LFP.

    Returns (coherence, freqs_hz). SFC = |S_xy|^2 / (S_xx * S_yy).
    """
    n = min(binary_train.size, lfp_signal.size)
    if n < 2:
        return np.array([]), np.array([])
    a = binary_train[:n].astype(np.float64) - binary_train[:n].mean()
    b = lfp_signal[:n].astype(np.float64) - lfp_signal[:n].mean()
    fa, fb = np.fft.rfft(a), np.fft.rfft(b)
    sab = fa * np.conj(fb)
    saa = np.abs(fa) ** 2
    sbb = np.abs(fb) ** 2
    denom = saa * sbb
    denom[denom == 0] = 1e-30
    sfc = np.abs(sab) ** 2 / denom
    return sfc, np.fft.rfftfreq(n, d=dt)


def spike_phase_histogram(
    binary_train: np.ndarray[Any, Any], lfp_signal: np.ndarray[Any, Any], n_bins: int = 36
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Histogram of LFP phase at spike times.

    Returns (counts, bin_centers_rad) with bins spanning [-pi, pi].
    """
    n = min(binary_train.size, lfp_signal.size)
    analytic = np.fft.ifft(
        np.fft.fft(lfp_signal[:n].astype(np.float64)) * 2 * (np.arange(n) > 0).astype(np.float64)
    )
    phase = np.angle(analytic)
    spike_phases = phase[binary_train[:n] > 0]
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    hist, _ = np.histogram(spike_phases, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    return hist, centers
