# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic cross-correlation diagnostics for SC bitstreams

r"""Measure stochastic cross-correlation (SCC) between two SC bitstreams.

Stochastic-computing ``AND`` multiplication assumes the operand bitstreams are
*independent*: :math:`E[\text{AND}] = p_a p_b`. Shared or overlapping random
sources break that assumption and bias the product. The stochastic
cross-correlation :math:`\text{SCC} \in [-1, 1]` (Alaghi & Hayes 2013) quantifies
the deviation from independence directly from the observed :math:`2\times2`
overlap counts of the two streams:

===============  ===============  ===============
                 :math:`b_B = 1`  :math:`b_B = 0`
===============  ===============  ===============
:math:`b_A = 1`  :math:`a`        :math:`b`
:math:`b_A = 0`  :math:`c`        :math:`d`
===============  ===============  ===============

With :math:`p_a = (a+b)/N`, :math:`p_b = (a+c)/N` and the independent overlap
count :math:`e = (a+b)(a+c)/N`,

.. math::
    \text{SCC} = \begin{cases}
        (a - e) / (\min(a+b,\,a+c) - e) & a \ge e \\
        (a - e) / (e - \max(0,\,a+b+a+c-N)) & a < e,
    \end{cases}

which is the exact inverse of :func:`sc_neurocore.core.sc_error_bounds.
multiply_correlation_bias`: the observed AND bias :math:`a/N - p_a p_b` equals the
bias that formula predicts from the measured SCC. This module is the diagnostic
half of the correlation-bias question flagged in the SC gap register — it lets a
design surface fail loud when re-convergent fan-ins share a source.

References
----------
Alaghi, A. & Hayes, J. P. (2013). Exposing correlation in stochastic computing.
(SCC definition and the overlap-count estimator.)
Alaghi, A. & Hayes, J. P. (2013). Survey of stochastic computing.
ACM TECS 12(2s):1-19.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .sc_error_bounds import multiply_correlation_bias

UInt8Array = NDArray[np.uint8]


def _as_bit_array(bits: Any, name: str) -> UInt8Array:
    array = np.asarray(bits, dtype=np.uint8)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only 0/1 bits")
    return array


def _overlap_counts(bits_a: UInt8Array, bits_b: UInt8Array) -> tuple[int, int, int, int]:
    a = int(np.sum((bits_a == 1) & (bits_b == 1)))
    b = int(np.sum((bits_a == 1) & (bits_b == 0)))
    c = int(np.sum((bits_a == 0) & (bits_b == 1)))
    d = int(np.sum((bits_a == 0) & (bits_b == 0)))
    return a, b, c, d


def estimate_scc(bits_a: Any, bits_b: Any) -> float:
    r"""Estimate the stochastic cross-correlation of two equal-length bitstreams.

    Parameters
    ----------
    bits_a, bits_b : array-like of {0, 1}
        Equal-length observed SC bitstreams.

    Returns
    -------
    float
        SCC in :math:`[-1, 1]`. Returns ``0.0`` for a degenerate stream (all-zero
        or all-one), where correlation is not identifiable.

    Raises
    ------
    ValueError
        If the streams are empty, non-binary, or of unequal length.
    """
    array_a = _as_bit_array(bits_a, "bits_a")
    array_b = _as_bit_array(bits_b, "bits_b")
    if array_a.size != array_b.size:
        raise ValueError("bits_a and bits_b must have equal length")

    n = array_a.size
    a, b, c, _d = _overlap_counts(array_a, array_b)
    ones_a = a + b
    ones_b = a + c
    expected = ones_a * ones_b / n
    if a >= expected:
        denom = min(ones_a, ones_b) - expected
    else:
        denom = expected - max(0, ones_a + ones_b - n)
    if denom == 0.0:
        return 0.0
    return float((a - expected) / denom)


def observed_and_bias(bits_a: Any, bits_b: Any) -> float:
    r"""Return the measured ``AND`` bias :math:`a/N - p_a p_b` of two streams.

    Parameters
    ----------
    bits_a, bits_b : array-like of {0, 1}
        Equal-length observed SC bitstreams.

    Returns
    -------
    float
        The signed deviation of the observed one-probability of ``AND`` from the
        independent product :math:`p_a p_b`.
    """
    array_a = _as_bit_array(bits_a, "bits_a")
    array_b = _as_bit_array(bits_b, "bits_b")
    if array_a.size != array_b.size:
        raise ValueError("bits_a and bits_b must have equal length")
    n = array_a.size
    a, b, c, _d = _overlap_counts(array_a, array_b)
    p_a = (a + b) / n
    p_b = (a + c) / n
    return a / n - p_a * p_b


@dataclass(frozen=True)
class CorrelationDiagnostic:
    """Result of a two-stream SC correlation check.

    Attributes
    ----------
    value_a, value_b : float
        Decoded one-probabilities of the two streams.
    scc : float
        Estimated stochastic cross-correlation in [-1, 1].
    predicted_and_bias : float
        AND bias the SCC implies via ``multiply_correlation_bias``.
    observed_and_bias : float
        AND bias measured directly from the streams.
    flagged : bool
        Whether ``abs(predicted_and_bias)`` exceeds the configured threshold.
    """

    value_a: float
    value_b: float
    scc: float
    predicted_and_bias: float
    observed_and_bias: float
    flagged: bool


def correlation_diagnostic(
    bits_a: Any, bits_b: Any, *, bias_threshold: float = 0.01
) -> CorrelationDiagnostic:
    r"""Diagnose ``AND``-composition correlation bias between two SC bitstreams.

    Estimates the SCC, converts it to the predicted AND bias through
    :func:`sc_neurocore.core.sc_error_bounds.multiply_correlation_bias`, compares
    it with the directly-measured bias, and flags streams whose predicted bias
    exceeds ``bias_threshold``.

    Parameters
    ----------
    bits_a, bits_b : array-like of {0, 1}
        Equal-length observed SC bitstreams.
    bias_threshold : float, optional
        Absolute predicted-bias magnitude above which the pair is flagged
        (default ``0.01``). Must be non-negative.

    Returns
    -------
    CorrelationDiagnostic
        Decoded values, SCC, predicted and observed AND bias, and the flag.
    """
    array_a = _as_bit_array(bits_a, "bits_a")
    array_b = _as_bit_array(bits_b, "bits_b")
    if array_a.size != array_b.size:
        raise ValueError("bits_a and bits_b must have equal length")
    threshold = float(bias_threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("bias_threshold must be non-negative")

    value_a = float(np.mean(array_a))
    value_b = float(np.mean(array_b))
    scc = estimate_scc(array_a, array_b)
    predicted = multiply_correlation_bias(value_a, value_b, scc)
    observed = observed_and_bias(array_a, array_b)
    return CorrelationDiagnostic(
        value_a=value_a,
        value_b=value_b,
        scc=scc,
        predicted_and_bias=predicted,
        observed_and_bias=observed,
        flagged=abs(predicted) > threshold,
    )
