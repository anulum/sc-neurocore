# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bitstream encoders and sliding-window probability estimation

"""Encode and decode stochastic bitstreams and estimate their sample probabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, SupportsInt

import numpy as np

from sc_neurocore._source_checkout import require_source_checkout_package
from sc_neurocore.exceptions import SCEncodingError

from .rng import RNG


def _require_finite_encoding_scalar(value: float, name: str) -> float:
    """Validate that ``value`` is a finite real scalar (not ``bool`` or complex)."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
    ):
        raise SCEncodingError(f"{name} must be a finite real number, got {value!r}.")
    return float(value)


def _require_non_negative_length(length: int) -> int:
    """Validate that ``length`` is a non-negative integer (not ``bool``)."""
    if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or int(length) < 0:
        raise SCEncodingError(f"Bitstream length must be a non-negative integer, got {length!r}.")
    return int(length)


def _require_finite_bitstream(bitstream: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Validate that ``bitstream`` is non-empty and contains only finite numeric values."""
    arr = np.asarray(bitstream)
    if arr.size == 0:
        raise SCEncodingError("Bitstream is empty.")
    if arr.dtype == np.object_ or np.iscomplexobj(arr) or not np.all(np.isfinite(arr)):
        raise SCEncodingError("Bitstream must contain only finite numeric values.")
    return arr


def generate_bernoulli_bitstream(
    p: float,
    length: int,
    rng: Optional[RNG] = None,
) -> np.ndarray[Any, Any]:
    """Generate a Bernoulli bitstream with probability ``p`` of a one.

    Parameters
    ----------
    p : float
        Probability of 1 (unipolar encoding, ``0 <= p <= 1``).
    length : int
        Number of bits in the stream.
    rng : RNG, optional
        RNG instance. If None, a fresh RNG is created.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(length,)`` with ``dtype=uint8`` and values in ``{0, 1}``.

    Raises
    ------
    SCEncodingError
        If ``p`` is boolean, non-finite, or outside ``[0, 1]``, or if ``length``
        is negative or boolean.
    """
    p_val = _require_finite_encoding_scalar(p, "Probability p")
    n_bits = _require_non_negative_length(length)
    if not 0.0 <= p_val <= 1.0:
        raise SCEncodingError(f"Probability p must be in [0,1], got {p_val}.")
    if rng is None:
        rng = RNG()
    encoded: np.ndarray[Any, Any] = np.asarray(rng.bernoulli(p_val, size=n_bits), dtype=np.uint8)
    return encoded


def generate_sobol_bitstream(
    p: float,
    length: int,
    seed: Optional[int] = None,
) -> np.ndarray[Any, Any]:
    """Generate a bitstream using a Sobol low-discrepancy sequence.

    Low-discrepancy sequences provide faster convergence than random Bernoulli
    sequences (``O(1/N)`` vs ``O(1/sqrt(N))``).

    Parameters
    ----------
    p : float
        Target probability in ``[0, 1]``.
    length : int
        Length of the bitstream.
    seed : int, optional
        Seed for the Sobol engine.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(length,)`` with ``dtype=uint8`` and values in ``{0, 1}``.

    Raises
    ------
    SCEncodingError
        If ``p`` is boolean, non-finite, or outside ``[0, 1]``, or if ``length``
        is negative or boolean.
    """
    p_val = _require_finite_encoding_scalar(p, "Probability p")
    n_bits = _require_non_negative_length(length)
    if not 0.0 <= p_val <= 1.0:
        raise SCEncodingError(f"Probability p must be in [0,1], got {p_val}.")

    import scipy.stats.qmc as qmc

    sampler = qmc.Sobol(d=1, seed=seed)
    samples = sampler.random(n=n_bits).flatten()
    bits: np.ndarray[Any, Any] = (samples < p_val).astype(np.uint8)
    return bits


def bitstream_to_probability(bitstream: np.ndarray[Any, Any]) -> float:
    """Decode a unipolar bitstream back into a probability estimate.

    Parameters
    ----------
    bitstream : numpy.ndarray
        Non-empty finite binary stream.

    Returns
    -------
    float
        Sample mean ``(# of ones) / length``.

    Raises
    ------
    SCEncodingError
        If ``bitstream`` is empty or contains non-finite values.
    """
    arr = _require_finite_bitstream(bitstream)
    return float(arr.mean())


def generate_bipolar_bitstream(
    x: float,
    length: int,
    rng: Optional[RNG] = None,
) -> np.ndarray[Any, Any]:
    """Generate a bipolar stochastic bitstream encoding a value in ``[-1, +1]``.

    Bipolar encoding maps ``x`` in ``[-1, 1]`` to probability ``p = (x + 1) / 2``;
    decoding recovers ``x = 2 * mean(bits) - 1``.

    Parameters
    ----------
    x : float
        Bipolar value in ``[-1, 1]``.
    length : int
        Number of bits in the stream.
    rng : RNG, optional
        RNG instance.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(length,)`` with ``dtype=uint8``.

    Raises
    ------
    SCEncodingError
        If ``x`` is boolean, non-finite, or outside ``[-1, 1]``, or if ``length``
        is negative or boolean.
    """
    x_val = _require_finite_encoding_scalar(x, "Bipolar value")
    if not -1.0 <= x_val <= 1.0:
        raise SCEncodingError(f"Bipolar value must be in [-1,1], got {x_val}.")
    p = (x_val + 1.0) / 2.0
    return generate_bernoulli_bitstream(p, length, rng)


def bipolar_to_value(bitstream: np.ndarray[Any, Any]) -> float:
    """Decode a bipolar bitstream to a value in ``[-1, +1]``.

    Parameters
    ----------
    bitstream : numpy.ndarray
        Non-empty finite binary stream.

    Returns
    -------
    float
        Decoded bipolar estimate ``2 * mean(bits) - 1``.

    Raises
    ------
    SCEncodingError
        If ``bitstream`` is empty or contains non-finite values.
    """
    arr = _require_finite_bitstream(bitstream)
    return float(2.0 * arr.mean() - 1.0)


def value_to_bipolar_prob(x: float) -> float:
    """Map a value in ``[-1, 1]`` to its unipolar bipolar-encoding probability.

    Parameters
    ----------
    x : float
        Bipolar value in ``[-1, 1]``.

    Returns
    -------
    float
        Probability ``(x + 1) / 2``.

    Raises
    ------
    SCEncodingError
        If ``x`` is boolean, non-finite, or outside ``[-1, 1]``.
    """
    x_val = _require_finite_encoding_scalar(x, "Bipolar value")
    if not -1.0 <= x_val <= 1.0:
        raise SCEncodingError(f"Bipolar value must be in [-1,1], got {x_val}.")
    return (x_val + 1.0) / 2.0


def value_to_unipolar_prob(
    x: float,
    x_min: float,
    x_max: float,
    clip: bool = True,
) -> float:
    """Map a finite scalar ``x`` from ``[x_min, x_max]`` into ``[0, 1]``.

    Parameters
    ----------
    x : float
        Finite input scalar.
    x_min : float
        Finite lower domain bound.
    x_max : float
        Finite upper domain bound (``x_min < x_max``).
    clip : bool, default=True
        When True, clamp finite ``x`` outside ``[x_min, x_max]`` to the domain
        endpoints; non-finite values are always rejected.

    Returns
    -------
    float
        Linearly mapped probability ``(x - x_min) / (x_max - x_min)``.

    Raises
    ------
    SCEncodingError
        If ``x``, ``x_min``, or ``x_max`` is boolean or non-finite, if
        ``x_min >= x_max``, or if ``clip=False`` and ``x`` lies outside
        ``[x_min, x_max]``.
    """
    x_val = _require_finite_encoding_scalar(x, "Input x")
    lo = _require_finite_encoding_scalar(x_min, "x_min")
    hi = _require_finite_encoding_scalar(x_max, "x_max")
    if lo >= hi:
        raise SCEncodingError("x_min must be < x_max.")
    if clip:
        x_val = max(min(x_val, hi), lo)
    elif not lo <= x_val <= hi:
        raise SCEncodingError(f"Input x must be in [{lo}, {hi}] when clip=False, got {x_val}.")
    return float((x_val - lo) / (hi - lo))


def unipolar_prob_to_value(
    p: float,
    x_min: float,
    x_max: float,
) -> float:
    """Map a unipolar probability ``p`` in ``[0, 1]`` back to ``[x_min, x_max]``.

    Parameters
    ----------
    p : float
        Unipolar probability in ``[0, 1]``.
    x_min : float
        Finite lower domain bound.
    x_max : float
        Finite upper domain bound (``x_min < x_max``).

    Returns
    -------
    float
        Reconstructed scalar ``x_min + p * (x_max - x_min)``.

    Raises
    ------
    SCEncodingError
        If ``p``, ``x_min``, or ``x_max`` is boolean or non-finite, if
        ``x_min >= x_max``, or if ``p`` lies outside ``[0, 1]``.
    """
    p_val = _require_finite_encoding_scalar(p, "Probability p")
    lo = _require_finite_encoding_scalar(x_min, "x_min")
    hi = _require_finite_encoding_scalar(x_max, "x_max")
    if lo >= hi:
        raise SCEncodingError("x_min must be < x_max.")
    if not 0.0 <= p_val <= 1.0:
        raise SCEncodingError(f"Probability p must be in [0,1], got {p_val}.")
    return float(lo + p_val * (hi - lo))


def adaptive_length(
    p: float,
    epsilon: float = 0.01,
    confidence: float = 0.95,
    method: str = "hoeffding",
    min_length: int = 64,
    max_length: int = 65536,
) -> int:
    """Compute minimum bitstream length for target precision.

    Given probability ``p`` and error tolerance ``epsilon``, returns the smallest
    power-of-two length ``L`` within ``[min_length, max_length]`` satisfying
    ``|p_hat - p| < epsilon`` at the requested confidence.

    Parameters
    ----------
    p : float
        Encoded probability in ``[0, 1]``.
    epsilon : float, default=0.01
        Maximum acceptable absolute error (strictly positive).
    confidence : float, default=0.95
        Confidence level in ``(0, 1)``.
    method : str, default='hoeffding'
        Bound type: ``"hoeffding"``, ``"chebyshev"``, or ``"variance"``.
    min_length : int, default=64
        Minimum returned length.
    max_length : int, default=65536
        Maximum returned length (hardware cap).

    Returns
    -------
    int
        Minimum bitstream length rounded up to the nearest power of two.

    Raises
    ------
    ValueError
        If ``p``, ``epsilon``, or ``confidence`` is boolean, non-finite, or
        outside its valid domain, or if ``method`` is unknown.
    """
    if (
        isinstance(p, bool)
        or not isinstance(p, (int, float, np.integer, np.floating))
        or not math.isfinite(float(p))
        or not 0.0 <= float(p) <= 1.0
    ):
        raise ValueError(f"p must be a finite probability in [0, 1], got {p!r}")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float, np.integer, np.floating))
        or not math.isfinite(float(epsilon))
        or float(epsilon) <= 0.0
    ):
        raise ValueError(f"epsilon must be finite and positive, got {epsilon!r}")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float, np.integer, np.floating))
        or not math.isfinite(float(confidence))
    ):
        raise ValueError(f"confidence must be a finite real number, got {confidence!r}")

    p_val = float(p)
    eps_val = float(epsilon)
    conf_val = float(confidence)

    if method == "variance":
        var_factor = p_val * (1.0 - p_val)
        L = var_factor / (eps_val**2)
    elif method == "chebyshev":
        delta = 1.0 - conf_val
        if delta <= 0.0 or conf_val <= 0.0:
            raise ValueError("confidence must be in (0.0, 1.0)")
        L = p_val * (1.0 - p_val) / (eps_val**2 * delta)
    elif method == "hoeffding":
        delta = 1.0 - conf_val
        if delta <= 0.0 or conf_val <= 0.0:
            raise ValueError("confidence must be in (0.0, 1.0)")
        L = -math.log(delta / 2.0) / (2.0 * eps_val**2)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hoeffding', 'chebyshev', or 'variance'.")

    L_int = max(min_length, int(math.ceil(L)))
    L_pow2 = 1
    while L_pow2 < L_int:
        L_pow2 *= 2
    return min(L_pow2, max_length)


def sc_divide(
    numerator: np.ndarray[Any, Any],
    denominator: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Stochastic computing division via the CORDIV circuit.

    T.-H. Chen and J. P. Hayes, "Design of division circuits for stochastic
    computing", IEEE ISVLSI 2016, doi:10.1109/ISVLSI.2016.48.

    Sequential circuit: at each bit position ``t``,
      - ``x[t]=1``         -> ``z[t] = 1``
      - ``x[t]=0, y[t]=1`` -> ``z[t] = 0``
      - ``x[t]=0, y[t]=0`` -> ``z[t] = z[t-1]`` (hold)

    Parameters
    ----------
    numerator : numpy.ndarray
        Bitstream (``uint8``, ``{0, 1}``) of length ``L``.
    denominator : numpy.ndarray
        Bitstream (``uint8``, ``{0, 1}``) of length ``L``, generated from the same
        random sequence as ``numerator`` with higher or equal density.

    Returns
    -------
    numpy.ndarray
        Quotient bitstream of length ``L``.
    """
    numerator = np.asarray(numerator, dtype=np.uint8)
    denominator = np.asarray(denominator, dtype=np.uint8)
    if numerator.shape != denominator.shape:
        raise ValueError("numerator and denominator must have the same shape")

    out = np.zeros_like(numerator)
    prev = 0
    for t in range(len(numerator)):
        if numerator[t] == 1:
            out[t] = 1
        elif denominator[t] == 1:
            out[t] = 0
        else:
            out[t] = prev
        prev = out[t]
    return out


@dataclass
class BitstreamEncoder:
    """Encode continuous finite scalar values into stochastic bitstreams.

    Parameters
    ----------
    x_min : float, default=0.0
        Finite lower bound of the scalar input domain.
    x_max : float, default=1.0
        Finite upper bound of the scalar input domain (``x_min < x_max``).
    length : int, default=256
        Positive number of bits per encoded stream.
    seed : int, optional
        Optional PRNG seed.
    mode : str, default='bernoulli'
        Encoding backend: ``"bernoulli"``, ``"sobol"``, ``"bipolar"``, or
        ``"chaotic"``.
    """

    x_min: float = 0.0
    x_max: float = 1.0
    length: int = 256
    seed: Optional[int] = None
    mode: str = "bernoulli"

    def __post_init__(self) -> None:
        """Validate domain bounds and length, and create the mode-specific RNG."""
        lo = _require_finite_encoding_scalar(self.x_min, "x_min")
        hi = _require_finite_encoding_scalar(self.x_max, "x_max")
        if lo >= hi:
            raise SCEncodingError("x_min must be < x_max.")
        if (
            isinstance(self.length, bool)
            or not isinstance(self.length, (int, np.integer))
            or int(self.length) <= 0
        ):
            raise SCEncodingError(f"length must be a positive integer, got {self.length!r}.")

        if self.mode in ("bernoulli", "bipolar"):
            self._rng = RNG(self.seed)
        elif self.mode == "chaotic":
            require_source_checkout_package(
                "sc_neurocore.chaos.rng", "BitstreamEncoder(mode='chaotic')"
            )
            from sc_neurocore.chaos.rng import ChaoticRNG

            x0 = (self.seed % 997) / 1000.0 + 0.001 if self.seed is not None else 0.5
            self._chaotic_rng = ChaoticRNG(r=4.0, x=x0)
        elif self.mode != "sobol":
            raise SCEncodingError(f"Unknown mode: {self.mode}")

    def encode(self, x: float) -> np.ndarray[Any, Any]:
        """Encode one finite scalar into a stochastic bitstream.

        Parameters
        ----------
        x : float
            Finite scalar to encode; finite values outside ``[x_min, x_max]``
            are clamped to the domain endpoints, whereas ``NaN``, ``±Inf``, and
            booleans raise :class:`SCEncodingError` without advancing RNG state.

        Returns
        -------
        numpy.ndarray
            Binary ``uint8`` bitstream of shape ``(length,)``.

        Raises
        ------
        SCEncodingError
            If ``x`` or the configured domain bounds are boolean or non-finite,
            or if ``x_min >= x_max``.
        """
        x_val = _require_finite_encoding_scalar(x, "Input x")
        lo = _require_finite_encoding_scalar(self.x_min, "x_min")
        hi = _require_finite_encoding_scalar(self.x_max, "x_max")
        if lo >= hi:
            raise SCEncodingError("x_min must be < x_max.")
        if self.mode == "bipolar":
            x_clipped = max(min(x_val, hi), lo)
            bipolar_val = 2.0 * (x_clipped - lo) / (hi - lo) - 1.0
            return generate_bipolar_bitstream(bipolar_val, self.length, rng=self._rng)
        p = value_to_unipolar_prob(x_val, lo, hi, clip=True)
        if self.mode == "sobol":
            return generate_sobol_bitstream(p, self.length, seed=self.seed)
        if self.mode == "chaotic":
            return self._chaotic_rng.generate_bitstream(p, self.length)
        return generate_bernoulli_bitstream(p, self.length, rng=self._rng)

    def decode(self, bitstream: np.ndarray[Any, Any]) -> float:
        """Decode a stochastic bitstream back into the configured value range.

        Parameters
        ----------
        bitstream : numpy.ndarray
            Non-empty finite binary stream.

        Returns
        -------
        float
            Decoded scalar in ``[x_min, x_max]``.
        """
        lo = _require_finite_encoding_scalar(self.x_min, "x_min")
        hi = _require_finite_encoding_scalar(self.x_max, "x_max")
        if lo >= hi:
            raise SCEncodingError("x_min must be < x_max.")
        if self.mode == "bipolar":
            bipolar_val = bipolar_to_value(bitstream)
            return float(lo + (bipolar_val + 1.0) / 2.0 * (hi - lo))
        p_hat = bitstream_to_probability(bitstream)
        return unipolar_prob_to_value(p_hat, lo, hi)


@dataclass
class BitstreamAverager:
    """Sliding-window probability estimator for bitstreams.

    Parameters
    ----------
    window : int
        Positive number of samples retained in the sliding window.

    Raises
    ------
    SCEncodingError
        If the window is zero or negative.
    MemoryError
        If storage for the window cannot be allocated.
    """

    window: int
    _buffer: Optional[np.ndarray[Any, Any]] = None
    _index: int = 0
    _filled: bool = False
    _running_sum: int = 0

    def __post_init__(self) -> None:
        """Validate the window and allocate its initially empty sample storage."""
        if self.window <= 0:
            raise SCEncodingError("window must be > 0")
        try:
            self._buffer = np.zeros(self.window, dtype=np.uint8)
        except MemoryError as error:
            raise MemoryError("Bitstream averager buffer cannot be allocated") from error
        self._running_sum = 0

    def push(self, bit: SupportsInt) -> None:
        """Add one binary sample to the sliding window.

        Parameters
        ----------
        bit : int or SupportsInt
            A zero or one sample, including NumPy integer scalars.

        Raises
        ------
        SCEncodingError
            If the sample is not zero or one; buffered observations are unchanged.
        """
        if isinstance(bit, bool) or bit not in (0, 1):
            raise SCEncodingError("Bit must be 0 or 1.")
        sample = int(bit)

        assert self._buffer is not None
        old_bit = int(self._buffer[self._index])
        self._buffer[self._index] = sample

        if self._filled:
            self._running_sum = int(self._running_sum) - old_bit + sample
        else:
            self._running_sum += sample

        self._index = (self._index + 1) % self.window
        if self._index == 0:
            self._filled = True

    def estimate(self) -> float:
        """Return the current sliding-window probability estimate."""
        if not self._filled:
            count = self._index
            if count == 0:
                return 0.0
            return float(self._running_sum) / count
        return float(self._running_sum) / self.window

    def reset(self) -> None:
        """Clear all buffered samples and reset the estimator state."""
        assert self._buffer is not None
        self._buffer.fill(0)
        self._index = 0
        self._filled = False
        self._running_sum = 0
