# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Power spectrum of a binary spike train

"""Power spectrum of a binary spike train."""

from __future__ import annotations

from typing import Any

import numpy as np


def power_spectrum(
    binary_train: np.ndarray[Any, Any], dt: float = 0.001
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Return the periodogram of a binary spike train and its frequencies.

    The train's mean is subtracted before the real FFT, so the zero-frequency
    bin holds no constant offset.

    Parameters
    ----------
    binary_train : numpy.ndarray
        One entry per time step.
    dt : float
        Time step in seconds.

    Returns
    -------
    tuple of numpy.ndarray
        ``(psd, freqs_hz)``: the squared FFT magnitude divided by the number of
        samples, and the matching non-negative frequencies in hertz. Both are
        empty when the train has fewer than two samples.
    """
    n = binary_train.size
    if n < 2:
        return np.array([]), np.array([])
    x = binary_train.astype(np.float64) - binary_train.mean()
    fft_vals = np.fft.rfft(x)
    psd = np.abs(fft_vals) ** 2 / n
    freqs = np.fft.rfftfreq(n, d=dt)
    return psd, freqs
