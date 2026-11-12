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
(Alaghi & Hayes 2013). A deterministic source replaces this with a worst-case
bound that belongs to the points actually used: the star discrepancy of that
prefix (:func:`threshold_encoding_error_bound`). It equals :math:`1/N` for the
first :math:`N = 2^k` points of the base-2 radical-inverse sequence
(:func:`low_discrepancy_error_bound`; Najafi, Lilja & Riedel 2018) and can be
larger for any other length. Gate composition then propagates these errors: an ``AND`` of two
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
Doerr, C., Gnewuch, M. & Wahlström, M. (2014). Calculation of discrepancy
measures and applications. In: A Panorama of Discrepancy Theory, Springer
(arXiv:1405.1653). (One-dimensional star discrepancy of a point set,
attributed there to Niederreiter.)
Hoeffding, W. (1963). Probability inequalities for sums of bounded random
variables. J. Amer. Statist. Assoc. 58(301):13-30.
Connolly, M. P. & Higham, N. J. (2025). Probabilistic error analysis of
limited-precision stochastic rounding. SIAM J. Sci. Comput.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


class GuaranteeNotDetermined(ValueError):
    """No proven bound exists for the requested configuration.

    Raised instead of returning a number when the assumption behind a closed
    form does not hold. It is a :class:`ValueError`, so existing handlers of
    invalid arguments also catch it.
    """


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


def _product_values(values_a: list[float], values_b: list[float]) -> list[float]:
    """Return the validated term products ``a_k * b_k`` of two equal-length vectors."""
    if len(values_a) != len(values_b):
        raise ValueError("values_a and values_b must have equal length")
    if not values_a:
        raise ValueError("dot product requires at least one term")
    products = []
    for index, (a, b) in enumerate(zip(values_a, values_b)):
        pa = _validate_probability(a, f"values_a[{index}]")
        pb = _validate_probability(b, f"values_b[{index}]")
        products.append(pa * pb)
    return products


def dot_product_variance(values_a: list[float], values_b: list[float], length: int) -> float:
    r"""Variance of an averaged unipolar dot product of two independent vectors.

    Every product stream :math:`\text{AND}(a_k, b_k)` is observed over all
    :math:`N` bits (one counter per term, or an adder tree) and the :math:`K`
    estimates are averaged. The scaled dot product
    :math:`\frac{1}{K}\sum_k a_k b_k` then has variance
    :math:`\frac{1}{K^2 N}\sum_k a_k b_k (1 - a_k b_k)`, whose standard error
    grows as :math:`O(1/\sqrt{K N})` — the SC analogue of the
    :math:`\sqrt{K}\,u` inner-product growth of Connolly & Higham (2025).

    This is **not** the variance of a multiplexer's single output stream,
    which sees only :math:`N` of the :math:`K N` product bits; use
    :func:`mux_dot_product_variance` or
    :func:`interleaved_dot_product_variance` for those circuits.

    Assumptions: all operand streams are independent, and each is a sequence
    of independent bits.

    Parameters
    ----------
    values_a, values_b : list of float
        Equal-length vectors of probabilities in :math:`[0, 1]`.
    length : int
        Bitstream length :math:`N` (positive).

    Returns
    -------
    float
        Variance of the averaged scaled dot-product estimate.
    """
    _validate_length(length)
    products = _product_values(values_a, values_b)
    k = len(products)
    return sum(q * (1.0 - q) for q in products) / (k * k * length)


def mux_dot_product_variance(values_a: list[float], values_b: list[float], length: int) -> float:
    r"""Variance of a dot product through a uniformly random multiplexer.

    One output stream of :math:`N` bits is formed; each bit is the bit of one
    product stream :math:`\text{AND}(a_k, b_k)`, chosen uniformly at random
    among the :math:`K` terms. The output bits are then independent with
    probability :math:`\bar q = \frac{1}{K}\sum_k a_k b_k`, and the estimate of
    :math:`\bar q` has variance :math:`\bar q (1 - \bar q) / N`.

    The selection itself adds variance: two constant terms 0 and 1 give
    :math:`1/(4N)`, where the averaged form gives zero.

    Assumptions: all operand streams are independent sequences of independent
    bits; the select input is uniform over the terms, independent from bit to
    bit and independent of the data.

    Parameters
    ----------
    values_a, values_b : list of float
        Equal-length vectors of probabilities in :math:`[0, 1]`.
    length : int
        Length :math:`N` of the multiplexer's output stream (positive).

    Returns
    -------
    float
        :math:`\bar q (1 - \bar q) / N`.
    """
    _validate_length(length)
    products = _product_values(values_a, values_b)
    mean = sum(products) / len(products)
    return mean * (1.0 - mean) / length


def interleaved_dot_product_variance(
    values_a: list[float], values_b: list[float], length: int
) -> float:
    r"""Variance of a dot product through a round-robin multiplexer.

    One output stream of :math:`N` bits is formed; bit :math:`t` is the bit of
    term :math:`t \bmod K`. When :math:`N` is a multiple of :math:`K` every
    term contributes :math:`N/K` bits, the estimate of
    :math:`\frac{1}{K}\sum_k a_k b_k` is unbiased, and its variance is
    :math:`\frac{1}{K N}\sum_k a_k b_k (1 - a_k b_k)`.

    When :math:`N` is not a multiple of :math:`K` the terms are not weighted
    equally and the estimate is biased; no variance is returned for that case.

    Assumptions: all operand streams are independent sequences of independent
    bits; the select input is the fixed sequence :math:`0, 1, \dots, K-1`
    repeated.

    Parameters
    ----------
    values_a, values_b : list of float
        Equal-length vectors of probabilities in :math:`[0, 1]`.
    length : int
        Length :math:`N` of the multiplexer's output stream; a positive
        multiple of the number of terms.

    Returns
    -------
    float
        Variance of the round-robin scaled dot-product estimate.

    Raises
    ------
    GuaranteeNotDetermined
        If ``length`` is not a multiple of the number of terms.
    """
    _validate_length(length)
    products = _product_values(values_a, values_b)
    k = len(products)
    if length % k:
        raise GuaranteeNotDetermined(
            "a round-robin multiplexer weights the terms equally only when the "
            "length is a multiple of the number of terms"
        )
    return sum(q * (1.0 - q) for q in products) / (k * length)


def threshold_encoding_error_bound(points: Sequence[float]) -> float:
    r"""Exact worst-case error of threshold encoding with a given set of points.

    A deterministic source encodes :math:`p \in [0, 1]` with comparator points
    :math:`x_1, \dots, x_N \in [0, 1)` as the bits :math:`[x_i < p]` and decodes
    :math:`\hat p = \#\{i : x_i < p\} / N`. Between two neighbouring points the
    count is constant, so the error is extreme at the points. With the points
    sorted, :math:`x_{(1)} \le \dots \le x_{(N)}`,

    .. math::
        \sup_{p \in [0, 1]} |\hat p - p|
        = \max_{1 \le i \le N} \max\!\left(\frac{i}{N} - x_{(i)},\;
          x_{(i)} - \frac{i - 1}{N}\right),

    which is the star discrepancy of the point set (Doerr, Gnewuch & Wahlström
    2014). The supremum is approached just above a point and need not be
    attained. The value is at least :math:`1/(2N)`.

    The bound belongs to these points: a different prefix of the same sequence
    has its own. It covers the encoding of one value. It says nothing about a
    gate that combines two streams, whose error depends on how the two point
    sets relate to each other.

    Parameters
    ----------
    points : sequence of float
        The comparator points actually used, in any order, each in
        :math:`[0, 1)`.

    Returns
    -------
    float
        The supremum of the decoding error over all :math:`p \in [0, 1]`.

    Raises
    ------
    ValueError
        If ``points`` is empty or holds a value that is not finite or lies
        outside :math:`[0, 1)`.
    """
    ordered = sorted(float(point) for point in points)
    count = len(ordered)
    if count == 0:
        raise ValueError("points must be non-empty")
    if not all(math.isfinite(point) for point in ordered):
        raise ValueError("points must be finite")
    if ordered[0] < 0.0 or ordered[-1] >= 1.0:
        raise ValueError("points must lie in [0, 1)")
    worst = 0.0
    for index, point in enumerate(ordered, start=1):
        worst = max(worst, index / count - point, point - (index - 1) / count)
    return worst


def low_discrepancy_error_bound(length: int) -> float:
    r"""Deterministic error bound :math:`1/N` of a power-of-two radical-inverse prefix.

    The first :math:`N = 2^k` points of the base-2 radical-inverse sequence
    (the unscrambled one-dimensional Sobol sequence, and the Halton sequence in
    base 2, both from index 0) are exactly :math:`\{0, 1/N, \dots, (N-1)/N\}`.
    Threshold encoding with them decodes :math:`\lceil pN \rceil / N`, so the
    error lies in :math:`[0, 1/N)` (Najafi, Lilja & Riedel 2018).

    For any other length the prefix is not that grid and the error can exceed
    :math:`1/N`: the first five points encode :math:`p = 0.3` as :math:`0.6`.
    No number is returned then. Use :func:`threshold_encoding_error_bound`
    with the points actually used; it is exact for every length, base and
    scrambling.

    Parameters
    ----------
    length : int
        Bitstream length :math:`N`; a positive power of two.

    Returns
    -------
    float
        :math:`1 / N`.

    Raises
    ------
    GuaranteeNotDetermined
        If ``length`` is not a power of two.
    """
    _validate_length(length)
    if length & (length - 1):
        raise GuaranteeNotDetermined(
            "the 1/N bound holds only when the length is a power of two; "
            "use threshold_encoding_error_bound with the points actually used"
        )
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
