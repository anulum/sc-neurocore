# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Spectral analysis of spike trains."""

from __future__ import annotations

import numpy as np


def power_spectrum(binary_train: np.ndarray, dt: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density of a binary spike train.

    Returns (psd, freqs_hz).
    """
    n = binary_train.size
    if n < 2:
        return np.array([]), np.array([])
    x = binary_train.astype(np.float64) - binary_train.mean()
    fft_vals = np.fft.rfft(x)
    psd = np.abs(fft_vals) ** 2 / n
    freqs = np.fft.rfftfreq(n, d=dt)
    return psd, freqs
