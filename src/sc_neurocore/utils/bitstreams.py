# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Generate a Bernoulli bitstream of given length with

from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass
import numpy as np
import scipy.stats.qmc as qmc
from .rng import RNG
from sc_neurocore.exceptions import SCEncodingError


def generate_bernoulli_bitstream(
    p: float,
    length: int,
    rng: Optional[RNG] = None,
) -> np.ndarray[Any, Any]:
    """
    Generate a Bernoulli bitstream of given length with probability p of '1'.
    This is the core SC primitive: a sequence of 0/1 bits where the
    proportion of 1s ~ p.
    Parameters
    ----------
    p : float
        Probability of 1 (unipolar encoding, 0 <= p <= 1).
    length : int
        Number of bits in the stream.
    rng : RNG, optional
        RNG instance. If None, a fresh RNG is created.
    Returns
    -------
    np.ndarray
        Array of shape (length,) with dtype=uint8, values in {0,1}.
    """
    if not 0.0 <= p <= 1.0:
        raise SCEncodingError(f"Probability p must be in [0,1], got {p}.")
    if rng is None:
        rng = RNG()
    bits = rng.bernoulli(p, size=length)
    return bits.astype(np.uint8)


def generate_sobol_bitstream(
    p: float,
    length: int,
    seed: Optional[int] = None,
) -> np.ndarray[Any, Any]:
    """
    Generate a bitstream using a Sobol sequence (Low Discrepancy Sequence).
    LDS provides faster convergence than random Bernoulli sequences (O(1/N) vs O(1/sqrt(N))).

    Parameters
    ----------
    p : float
        Target probability.
    length : int
        Length of the bitstream.
    seed : int, optional
        Seed for the Sobol engine.

    Returns
    -------
    np.ndarray
        Array of shape (length,) with dtype=uint8, values in {0,1}.
    """
    if not 0.0 <= p <= 1.0:
        raise SCEncodingError(f"Probability p must be in [0,1], got {p}.")

    # Create Sobol engine (1 dimension)
    sampler = qmc.Sobol(d=1, seed=seed)

    # Generate samples. Sobol works best with powers of 2,
    # but we can take 'length' samples.
    # Note: For strict determinism, one should manage the sampler state,
    # but here we create a fresh one or seek could be used if persisting.
    # To avoid 'scramble' creating randomness if not desired, we set scramble=False by default in Sobol,
    # but scramble=True usually gives better results for integration-like tasks.
    # We'll use scramble=True with the seed.

    # Optimally, length should be power of 2 for Sobol balance properties.
    # We allow any length but warn or just proceed.

    samples = sampler.random(n=length)  # Shape (length, 1)
    samples = samples.flatten()

    # Thresholding: The standard way to convert a U[0,1] sample 's' to a bit with prob 'p'
    # is: bit = 1 if s < p else 0
    bits = (samples < p).astype(np.uint8)

    return bits


def bitstream_to_probability(bitstream: np.ndarray[Any, Any]) -> float:
    """
    Decode a unipolar bitstream back into a probability estimate.
    p_hat = (# of ones) / length
    """
    if bitstream.size == 0:
        raise SCEncodingError("Bitstream is empty.")
    return float(bitstream.mean())


def generate_bipolar_bitstream(
    x: float,
    length: int,
    rng: Optional[RNG] = None,
) -> np.ndarray[Any, Any]:
    """Generate a bipolar SC bitstream encoding a value in [-1, +1].

    Bipolar encoding: value x in [-1, 1] maps to probability p = (x + 1) / 2.
    Bit=1 with probability p, bit=0 with probability 1-p.
    Decoding: x = 2 * mean(bits) - 1.

    Bipolar multiplication uses XNOR: P(A XNOR B) encodes A*B in bipolar.
    """
    if not -1.0 <= x <= 1.0:
        raise SCEncodingError(f"Bipolar value must be in [-1,1], got {x}.")
    p = (x + 1.0) / 2.0
    return generate_bernoulli_bitstream(p, length, rng)


def bipolar_to_value(bitstream: np.ndarray[Any, Any]) -> float:
    """Decode a bipolar bitstream to a value in [-1, +1].

    x = 2 * mean(bits) - 1
    """
    if bitstream.size == 0:
        raise SCEncodingError("Bitstream is empty.")
    return float(2.0 * bitstream.mean() - 1.0)


def value_to_bipolar_prob(x: float) -> float:
    """Map a value in [-1, 1] to the unipolar probability used in bipolar encoding.

    p = (x + 1) / 2. This p is then used with standard Bernoulli generation.
    """
    if not -1.0 <= x <= 1.0:
        raise SCEncodingError(f"Bipolar value must be in [-1,1], got {x}.")
    return (x + 1.0) / 2.0


def value_to_unipolar_prob(
    x: float,
    x_min: float,
    x_max: float,
    clip: bool = True,
) -> float:
    """
    Map a scalar x from [x_min, x_max] into a unipolar probability [0,1].
    Linear mapping:
        p = (x - x_min) / (x_max - x_min)
    If clip=True, x is clipped into [x_min, x_max].
    """
    if x_min >= x_max:
        raise SCEncodingError("x_min must be < x_max.")
    if clip:
        x = max(min(x, x_max), x_min)
    p = (x - x_min) / (x_max - x_min)
    return float(p)


def unipolar_prob_to_value(
    p: float,
    x_min: float,
    x_max: float,
) -> float:
    """
    Map a unipolar probability p in [0,1] back to a scalar in [x_min, x_max].
    Inverse of value_to_unipolar_prob.
    """
    if not 0.0 <= p <= 1.0:
        raise SCEncodingError(f"Probability p must be in [0,1], got {p}.")
    return float(x_min + p * (x_max - x_min))


@dataclass
class BitstreamEncoder:
    """
    Helper for encoding continuous scalar values into SC bitstreams
    using linear unipolar mapping.
    Example
    -------
    encoder = BitstreamEncoder(x_min=0.0, x_max=0.1, length=1024, seed=123)
    bitstream = encoder.encode(0.06)  # 60% ones
    p_hat = bitstream_to_probability(bitstream)
    x_rec = encoder.decode(bitstream)
    """

    x_min: float
    x_max: float
    length: int = 256
    seed: Optional[int] = None
    mode: str = "bernoulli"  # "bernoulli", "sobol", or "bipolar"

    def __post_init__(self) -> None:
        if self.mode in ("bernoulli", "bipolar"):
            self._rng = RNG(self.seed)
        elif self.mode != "sobol":
            raise SCEncodingError(f"Unknown mode: {self.mode}")

    def encode(self, x: float) -> np.ndarray[Any, Any]:
        if self.mode == "bipolar":
            # Map x from [x_min, x_max] to [-1, 1], then bipolar encode
            if self.x_min >= self.x_max:
                raise SCEncodingError("x_min must be < x_max.")
            x_clipped = max(min(x, self.x_max), self.x_min)
            bipolar_val = 2.0 * (x_clipped - self.x_min) / (self.x_max - self.x_min) - 1.0
            return generate_bipolar_bitstream(bipolar_val, self.length, rng=self._rng)
        p = value_to_unipolar_prob(x, self.x_min, self.x_max, clip=True)
        if self.mode == "sobol":
            return generate_sobol_bitstream(p, self.length, seed=self.seed)
        return generate_bernoulli_bitstream(p, self.length, rng=self._rng)

    def decode(self, bitstream: np.ndarray[Any, Any]) -> float:
        if self.mode == "bipolar":
            bipolar_val = bipolar_to_value(bitstream)
            # Map [-1, 1] back to [x_min, x_max]
            return float(self.x_min + (bipolar_val + 1.0) / 2.0 * (self.x_max - self.x_min))
        p_hat = bitstream_to_probability(bitstream)
        return unipolar_prob_to_value(p_hat, self.x_min, self.x_max)


@dataclass
class BitstreamAverager:
    """
    Sliding-window probability estimator for bitstreams.

    Example
    -------
    >>> avg = BitstreamAverager(window=100)
    >>> for _ in range(100):
    ...     avg.push(1)
    >>> avg.estimate()
    1.0
    >>> avg.push(0)
    >>> avg.estimate() < 1.0
    True
    """

    window: int
    _buffer: Optional[np.ndarray[Any, Any]] = None
    _index: int = 0
    _filled: bool = False
    _running_sum: int = 0

    def __post_init__(self) -> None:
        self._buffer = np.zeros(self.window, dtype=np.uint8)
        self._running_sum = 0

    def push(self, bit: int) -> None:
        if bit not in (0, 1):
            raise SCEncodingError("Bit must be 0 or 1.")

        # Remove old bit from sum if buffer is wrapping around
        old_bit = self._buffer[self._index]
        self._buffer[self._index] = bit

        if self._filled:
            self._running_sum = self._running_sum - old_bit + bit
        else:
            self._running_sum += bit

        self._index = (self._index + 1) % self.window
        if self._index == 0:
            self._filled = True

    def estimate(self) -> float:
        if not self._filled:
            # Estimate over the filled portion only
            count = self._index
            if count == 0:
                return 0.0
            return float(self._running_sum) / count
        return float(self._running_sum) / self.window

    def reset(self) -> None:
        self._buffer.fill(0)  # type: ignore
        self._index = 0
        self._filled = False
        self._running_sum = 0
