# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reproducible stochastic-threshold primitive

"""One canonical LFSR16 Bernoulli contract for software and emitted RTL."""

from __future__ import annotations

import math
import secrets

DEFAULT_LFSR16_SEED = 0xACE1
LFSR16_PERIOD = 0xFFFF
LFSR16_THRESHOLD_FULL_SCALE = 0x10000
LFSR16_ADVANCES_PER_TRIAL = 8


def normalise_lfsr16_seed(seed: int | None) -> int:
    """Return a valid non-zero 16-bit seed.

    ``None`` requests independent entropy for ordinary Python model instances.
    Explicit zero uses the documented hardware fallback seed so that a C ABI or
    RTL parameter can never lock the maximal-length recurrence in the all-zero
    state.
    """
    if seed is None:
        return secrets.randbelow(LFSR16_PERIOD) + 1
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer in [0, 65535] or None")
    if not 0 <= seed <= LFSR16_PERIOD:
        raise ValueError("seed must be an integer in [0, 65535] or None")
    return DEFAULT_LFSR16_SEED if seed == 0 else seed


def lfsr16_advance(state: int) -> int:
    """Advance the canonical right-shift x^16+x^14+x^13+x^11+1 LFSR."""
    if isinstance(state, bool) or not isinstance(state, int) or not 1 <= state <= LFSR16_PERIOD:
        raise ValueError("LFSR16 state must be a non-zero 16-bit integer")
    feedback = ((state >> 0) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1
    return ((state >> 1) | (feedback << 15)) & LFSR16_PERIOD


def lfsr16_trial_sample(state: int) -> int:
    """Return the decimated sample used by one Bernoulli trial.

    Eight primitive advances suppress the strong adjacent-state correlation of
    a raw shift-register word. Eight is coprime with the 65,535-state period,
    so decimation retains the complete non-zero state cycle and exact rate
    quantisation rather than shortening the generator period.
    """
    for _ in range(LFSR16_ADVANCES_PER_TRIAL):
        state = lfsr16_advance(state)
    return state


def probability_to_lfsr16_threshold(probability: float) -> int:
    """Map ``p`` to an unbiased comparator threshold over 65,535 states.

    A trial advances the LFSR first and compares its non-zero sample with the
    returned 17-bit threshold. For ``0 < p < 1`` the realised probability is
    ``floor(p * 65535) / 65535``; the absolute quantisation error is therefore
    strictly below one LFSR period quantum. Zero never fires and one always
    fires, while both still consume one RNG sample.
    """
    probability = float(probability)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be finite and in [0, 1]")
    if probability <= 0.0:
        return 0
    if probability >= 1.0:
        return LFSR16_THRESHOLD_FULL_SCALE
    return math.floor(probability * LFSR16_PERIOD) + 1


class Lfsr16Threshold:
    """Stateful advance-before-compare Bernoulli sampler with replayable reset."""

    __slots__ = ("_initial_seed", "_state")

    def __init__(self, seed: int | None = DEFAULT_LFSR16_SEED) -> None:
        self._initial_seed = normalise_lfsr16_seed(seed)
        self._state = self._initial_seed

    @property
    def initial_seed(self) -> int:
        """Return the normalised seed restored by :meth:`reset`."""
        return self._initial_seed

    @property
    def state(self) -> int:
        """Return the last emitted LFSR sample/state."""
        return self._state

    def trial(self, probability: float) -> bool:
        """Take one eight-advance sample and perform one quantised trial."""
        threshold = probability_to_lfsr16_threshold(probability)
        self._state = lfsr16_trial_sample(self._state)
        return self._state < threshold

    def restore(self, state: int) -> None:
        """Restore a validated live state returned by a native backend."""
        if isinstance(state, bool) or not isinstance(state, int) or not 1 <= state <= LFSR16_PERIOD:
            raise ValueError("LFSR16 state must be a non-zero 16-bit integer")
        self._state = state

    def reset(self) -> None:
        """Restore the exact explicit or entropy-derived initial seed."""
        self._state = self._initial_seed
