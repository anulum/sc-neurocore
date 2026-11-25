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

from .basic import _require_finite_train, _require_positive_dt


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
    dt : float, default=0.001
        Strictly positive time step in seconds.

    Returns
    -------
    tuple of numpy.ndarray
        ``(psd, freqs_hz)``: the squared FFT magnitude divided by the number of
        samples, and the matching non-negative frequencies in hertz. Both are
        empty when the train has fewer than two samples.

    Raises
    ------
    ValueError
        If ``binary_train`` contains non-finite values or ``dt`` is boolean,
        non-finite, or not positive.
    """
    arr = _require_finite_train(binary_train)
    step_s = _require_positive_dt(dt)
    n = arr.size
    if n < 2:
        return np.array([]), np.array([])
    x = arr.astype(np.float64) - arr.mean()
    fft_vals = np.fft.rfft(x)
    psd = np.abs(fft_vals) ** 2 / n
    freqs = np.fft.rfftfreq(n, d=step_s)
    return psd, freqs
