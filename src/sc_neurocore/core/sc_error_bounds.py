# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Analytic error bounds for stochastic-computing arithmetic

r"""Closed-form accuracy bounds for stochastic-computing (SC) bitstream arithmetic.

A value :math:`p \in [0, 1]` is encoded as a length-:math:`N` Bernoulli bitstream
(``encoding/encoders.rate_encode`` semantics) and decoded as the sample mean
:math:`\hat p = k / N`. The estimator is unbiased with variance

.. math:: \operatorname{Var}(\hat p) = \frac{p (1 - p)}{N},

so the pseudo-random SC standard error shrinks as :math:`O(1/\sqrt N)`
(Alaghi & Hayes 2013). Deterministic low-discrepancy sources (Sobol/Halton)
replace this with an :math:`O(1/N)` deterministic bound (Najafi, Lilja &
Riedel 2018). Gate composition then propagates these errors: an ``AND`` of two
*independent* unipolar streams multiplies the values, but *correlated* streams
introduce a bias governed by the stochastic cross-correlation (SCC).

This module gives the analytic bounds for the common SC operations plus two
inverse "how long a bitstream do I need" calculators (variance-target and a
distribution-free Hoeffding confidence target). It is a closed-form analysis
surface — evaluated once per design query, not a per-timestep compute kernel —
so it is Python-only by design (no polyglot hot-path counterpart).

References
----------
Alaghi, A. & Hayes, J. P. (2013). Survey of stochastic computing.
ACM TECS 12(2s):1-19.
Alaghi, A. & Hayes, J. P. (2013). Exposing correlation in stochastic computing.
(SCC definition for correlated bitstream composition.)
Najafi, M. H., Lilja, D. J. & Riedel, M. (2018). Deterministic methods for
stochastic computing using low-discrepancy sequences. ICCAD.
Hoeffding, W. (1963). Probability inequalities for sums of bounded random
variables. J. Amer. Statist. Assoc. 58(301):13-30.
Connolly, M. P. & Higham, N. J. (2025). Probabilistic error analysis of
limited-precision stochastic rounding. SIAM J. Sci. Comput.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _validate_length(length: int) -> None:
    if isinstance(length, bool) or not isinstance(length, int):
        raise TypeError("length must be an int")
    if length <= 0:
        raise ValueError("bitstream length must be positive")


def _validate_probability(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return numeric


def _validate_bipolar(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric < -1.0 or numeric > 1.0:
        raise ValueError(f"{name} must lie in [-1, 1]")
    return numeric


def bernoulli_variance(value: float, length: int) -> float:
    r"""Variance of the decoded unipolar SC estimate :math:`\hat p = k/N`.

    Parameters
    ----------
    value : float
        The encoded probability :math:`p \in [0, 1]`.
    length : int
        Bitstream length :math:`N` (positive).

    Returns
    -------
    float
        :math:`p (1 - p) / N`.
    """
    p = _validate_probability(value, "value")
    _validate_length(length)
    return p * (1.0 - p) / length


def bernoulli_std_error(value: float, length: int) -> float:
    r"""Return the standard error :math:`\sqrt{p(1-p)/N}` of a unipolar SC estimate."""
    return math.sqrt(bernoulli_variance(value, length))


def bipolar_variance(value: float, length: int) -> float:
    r"""Variance of the decoded bipolar SC estimate for :math:`v \in [-1, 1]`.

    With :math:`p = (v + 1)/2` and :math:`\hat v = 2 \hat p - 1`,
    :math:`\operatorname{Var}(\hat v) = 4 \operatorname{Var}(\hat p)
    = (1 - v^2)/N`.

    Parameters
    ----------
    value : float
        Bipolar value :math:`v \in [-1, 1]`.
    length : int
        Bitstream length :math:`N` (positive).

    Returns
    -------
    float
        :math:`(1 - v^2)/N`.
    """
    v = _validate_bipolar(value, "value")
    _validate_length(length)
    return (1.0 - v * v) / length


def bipolar_std_error(value: float, length: int) -> float:
    r"""Return the standard error :math:`\sqrt{(1 - v^2)/N}` of a bipolar SC estimate."""
    return math.sqrt(bipolar_variance(value, length))


def multiply_variance(value_a: float, value_b: float, length: int) -> float:
    r"""Variance of a unipolar ``AND`` product of two *independent* streams.

    The output stream is Bernoulli with value :math:`p_a p_b`, so its decoded
    estimate has variance :math:`p_a p_b (1 - p_a p_b) / N`.

    Parameters
    ----------
    value_a, value_b : float
        Encoded probabilities in :math:`[0, 1]`.
    length : int
        Bitstream length :math:`N` (positive).

    Returns
    -------
    float
        :math:`p_a p_b (1 - p_a p_b) / N`.
    """
    pa = _validate_probability(value_a, "value_a")
    pb = _validate_probability(value_b, "value_b")
    _validate_length(length)
    product = pa * pb
    return product * (1.0 - product) / length


def multiply_correlation_bias(value_a: float, value_b: float, scc: float) -> float:
    r"""Signed bias of a unipolar ``AND`` from stochastic cross-correlation.

    For independent streams :math:`E[\text{AND}] = p_a p_b`. With stochastic
    cross-correlation :math:`\rho \in [-1, 1]` (Alaghi & Hayes), the joint
    one-probability moves linearly toward its comonotone bound
    :math:`\min(p_a, p_b)` for :math:`\rho > 0` and its countermonotone bound
    :math:`\max(0, p_a + p_b - 1)` for :math:`\rho < 0`:

    .. math::
        E[\text{AND}] = p_a p_b + \begin{cases}
            \rho\,(\min(p_a, p_b) - p_a p_b) & \rho \ge 0 \\
            \rho\,(p_a p_b - \max(0, p_a + p_b - 1)) & \rho < 0.
        \end{cases}

    Parameters
    ----------
    value_a, value_b : float
        Encoded probabilities in :math:`[0, 1]`.
    scc : float
        Stochastic cross-correlation :math:`\rho \in [-1, 1]`.

    Returns
    -------
    float
        The signed bias :math:`E[\text{AND}] - p_a p_b`.
    """
    pa = _validate_probability(value_a, "value_a")
    pb = _validate_probability(value_b, "value_b")
    rho = _validate_bipolar(scc, "scc")
    independent = pa * pb
    if rho >= 0.0:
        return rho * (min(pa, pb) - independent)
    return rho * (independent - max(0.0, pa + pb - 1.0))


def mux_add_variance(value_a: float, value_b: float, length: int) -> float:
    r"""Variance of a fair-select ``MUX`` scaled adder of two streams.

    A 2:1 multiplexer with select :math:`s \sim \mathrm{Bernoulli}(1/2)` emits a
    stream of value :math:`q = (p_a + p_b)/2`, whose estimate has variance
    :math:`q (1 - q) / N`.

    Parameters
    ----------
    value_a, value_b : float
        Encoded probabilities in :math:`[0, 1]`.
    length : int
        Bitstream length :math:`N` (positive).

    Returns
    -------
    float
        :math:`q (1 - q) / N` with :math:`q = (p_a + p_b)/2`.
    """
    pa = _validate_probability(value_a, "value_a")
    pb = _validate_probability(value_b, "value_b")
    _validate_length(length)
    q = 0.5 * (pa + pb)
    return q * (1.0 - q) / length


def dot_product_variance(values_a: list[float], values_b: list[float], length: int) -> float:
    r"""Variance of a MUX-tree unipolar dot product of two independent vectors.

    The scaled dot product :math:`\frac{1}{K}\sum_k \text{AND}(a_k, b_k)` of
    length :math:`K` has variance
    :math:`\frac{1}{K^2 N}\sum_k a_k b_k (1 - a_k b_k)`, whose standard error
    grows as :math:`O(1/\sqrt{K N})` — the SC analogue of the
    :math:`\sqrt{K}\,u` inner-product growth of Connolly & Higham (2025).

    Parameters
    ----------
    values_a, values_b : list of float
        Equal-length vectors of probabilities in :math:`[0, 1]`.
    length : int
        Bitstream length :math:`N` (positive).

    Returns
    -------
    float
        Variance of the scaled dot-product estimate.
    """
    _validate_length(length)
    if len(values_a) != len(values_b):
        raise ValueError("values_a and values_b must have equal length")
    if not values_a:
        raise ValueError("dot product requires at least one term")
    k = len(values_a)
    total = 0.0
    for index, (a, b) in enumerate(zip(values_a, values_b)):
        pa = _validate_probability(a, f"values_a[{index}]")
        pb = _validate_probability(b, f"values_b[{index}]")
        product = pa * pb
        total += product * (1.0 - product)
    return total / (k * k * length)


def low_discrepancy_error_bound(length: int) -> float:
    r"""Deterministic error bound :math:`1/N` for a low-discrepancy SC source.

    Sobol/Halton bitstreams are deterministic: a value representable at
    resolution :math:`1/N` is encoded exactly, and any value is within
    :math:`1/N`. This replaces the pseudo-random :math:`O(1/\sqrt N)` standard
    error with a hard :math:`O(1/N)` bound (Najafi, Lilja & Riedel 2018).

    Parameters
    ----------
    length : int
        Bitstream length :math:`N` (positive).

    Returns
    -------
    float
        :math:`1 / N`.
    """
    _validate_length(length)
    return 1.0 / length


def hoeffding_confidence(length: int, epsilon: float) -> float:
    r"""Distribution-free confidence that :math:`|\hat p - p| < \epsilon`.

    Hoeffding's inequality gives
    :math:`P(|\hat p - p| \ge \epsilon) \le 2 e^{-2 N \epsilon^2}`, so the
    confidence is :math:`\max(0, 1 - 2 e^{-2 N \epsilon^2})`.

    Parameters
    ----------
    length : int
        Bitstream length :math:`N` (positive).
    epsilon : float
        Absolute error tolerance in :math:`(0, 1]`.

    Returns
    -------
    float
        The confidence lower bound in :math:`[0, 1]`.
    """
    _validate_length(length)
    eps = float(epsilon)
    if not math.isfinite(eps) or eps <= 0.0 or eps > 1.0:
        raise ValueError("epsilon must lie in (0, 1]")
    return max(0.0, 1.0 - 2.0 * math.exp(-2.0 * length * eps * eps))


def hoeffding_min_length(epsilon: float, confidence: float) -> int:
    r"""Smallest :math:`N` guaranteeing ``confidence`` at tolerance ``epsilon``.

    Inverting Hoeffding's bound,
    :math:`N \ge \lceil \ln(2 / (1 - c)) / (2 \epsilon^2) \rceil`.

    Parameters
    ----------
    epsilon : float
        Absolute error tolerance in :math:`(0, 1]`.
    confidence : float
        Target confidence :math:`c \in [0, 1)`.

    Returns
    -------
    int
        The minimum distribution-free bitstream length.
    """
    eps = float(epsilon)
    if not math.isfinite(eps) or eps <= 0.0 or eps > 1.0:
        raise ValueError("epsilon must lie in (0, 1]")
    c = float(confidence)
    if not math.isfinite(c) or c < 0.0 or c >= 1.0:
        raise ValueError("confidence must lie in [0, 1)")
    return math.ceil(math.log(2.0 / (1.0 - c)) / (2.0 * eps * eps))


def min_length_for_std_error(value: float, target_std: float, *, bipolar: bool = False) -> int:
    r"""Smallest :math:`N` whose SC standard error is at most ``target_std``.

    Unipolar: :math:`N \ge \lceil p(1-p) / \sigma^2 \rceil`; bipolar:
    :math:`N \ge \lceil (1 - v^2) / \sigma^2 \rceil`.

    Parameters
    ----------
    value : float
        Encoded value: :math:`p \in [0, 1]` (unipolar) or :math:`v \in [-1, 1]`
        (bipolar).
    target_std : float
        Target standard error :math:`\sigma > 0`.
    bipolar : bool, optional
        Interpret ``value`` in the bipolar domain (default ``False``).

    Returns
    -------
    int
        The minimum bitstream length (at least 1).
    """
    sigma = float(target_std)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("target_std must be positive")
    if bipolar:
        v = _validate_bipolar(value, "value")
        numerator = 1.0 - v * v
    else:
        p = _validate_probability(value, "value")
        numerator = p * (1.0 - p)
    return max(1, math.ceil(numerator / (sigma * sigma)))


@dataclass(frozen=True)
class SCErrorBound:
    r"""Summary of the accuracy of one encoded SC value.

    Attributes
    ----------
    value : float
        The encoded value.
    length : int
        Bitstream length :math:`N`.
    bipolar : bool
        Whether the value is in the bipolar domain.
    variance : float
        Estimator variance.
    std_error : float
        Estimator standard error :math:`\sqrt{\text{variance}}`.
    ci95_halfwidth : float
        Half-width of the 95% normal-approximation confidence interval
        (:math:`1.959964 \times \text{std\_error}`).
    """

    value: float
    length: int
    bipolar: bool
    variance: float
    std_error: float
    ci95_halfwidth: float


_Z95 = 1.959963984540054  # two-sided 95% standard-normal quantile


def sc_error_bound(value: float, length: int, *, bipolar: bool = False) -> SCErrorBound:
    r"""Summarise the SC accuracy of one encoded value as an :class:`SCErrorBound`.

    Parameters
    ----------
    value : float
        Encoded value: :math:`p \in [0, 1]` (unipolar) or :math:`v \in [-1, 1]`
        (bipolar).
    length : int
        Bitstream length :math:`N` (positive).
    bipolar : bool, optional
        Interpret ``value`` in the bipolar domain (default ``False``).

    Returns
    -------
    SCErrorBound
        Variance, standard error, and 95% confidence half-width.
    """
    variance = bipolar_variance(value, length) if bipolar else bernoulli_variance(value, length)
    std_error = math.sqrt(variance)
    return SCErrorBound(
        value=float(value),
        length=length,
        bipolar=bipolar,
        variance=variance,
        std_error=std_error,
        ci95_halfwidth=_Z95 * std_error,
    )
