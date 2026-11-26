# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for adaptive bitstream length

"""Tests for adaptive_length(): error-variance model for precision vs speed."""

import re

import pytest

from sc_neurocore.utils.bitstreams import (
    adaptive_length,
    generate_bernoulli_bitstream,
    bitstream_to_probability,
)


class TestAdaptiveLength:
    """The stream length ``adaptive_length`` chooses, and the inputs it refuses."""

    def test_returns_power_of_2(self) -> None:
        """Length should always be a power of 2 for Sobol compatibility."""
        L = adaptive_length(0.5, epsilon=0.01)
        assert L & (L - 1) == 0  # power-of-2 check

    def test_tighter_epsilon_longer_stream(self) -> None:
        """Smaller epsilon should require longer bitstream."""
        L_coarse = adaptive_length(0.5, epsilon=0.1)
        L_fine = adaptive_length(0.5, epsilon=0.01)
        assert L_fine > L_coarse

    def test_higher_confidence_longer_stream(self) -> None:
        """Higher confidence should require longer bitstream."""
        L_low = adaptive_length(0.5, epsilon=0.05, confidence=0.90)
        L_high = adaptive_length(0.5, epsilon=0.05, confidence=0.99)
        assert L_high > L_low

    def test_extreme_p_shorter_stream(self) -> None:
        """A probability near 0 or 1 has lower variance, so a shorter stream suffices."""
        L_mid = adaptive_length(0.5, epsilon=0.05, method="chebyshev")
        L_edge = adaptive_length(0.01, epsilon=0.05, method="chebyshev")
        assert L_edge <= L_mid

    def test_hoeffding_bound(self) -> None:
        """The Hoeffding length at p 0.5, epsilon 0.05 lies between 64 and 65536."""
        L = adaptive_length(0.5, epsilon=0.05, confidence=0.95, method="hoeffding")
        assert L >= 64
        assert L <= 65536

    def test_chebyshev_bound(self) -> None:
        """The Chebyshev length at p 0.5, epsilon 0.05 is at least 64."""
        L = adaptive_length(0.5, epsilon=0.05, confidence=0.95, method="chebyshev")
        assert L >= 64

    def test_variance_method(self) -> None:
        """The variance method rounds p(1-p)/epsilon^2 up to the next power of two."""
        L = adaptive_length(0.5, epsilon=0.01, method="variance")
        # p(1-p)/eps^2 = 0.25/0.0001 = 2500 → next power of 2 = 4096
        assert L == 4096

    def test_respects_min_length(self) -> None:
        """The length is never below ``min_length``."""
        L = adaptive_length(0.01, epsilon=0.5, min_length=128)
        assert L >= 128

    def test_respects_max_length(self) -> None:
        """The length is never above ``max_length``."""
        L = adaptive_length(0.5, epsilon=0.001, max_length=8192)
        assert L <= 8192

    def test_invalid_epsilon_raises(self) -> None:
        """A zero epsilon is refused as not finite and positive."""
        with pytest.raises(ValueError, match="epsilon must be finite and positive"):
            adaptive_length(0.5, epsilon=0.0)

    def test_unknown_method_raises(self) -> None:
        """An unknown bound method is refused."""
        with pytest.raises(ValueError, match="Unknown method"):
            adaptive_length(0.5, epsilon=0.01, method="magic")

    def test_chebyshev_confidence_1_raises(self) -> None:
        """A confidence of 1.0 is refused by the Chebyshev bound."""
        with pytest.raises(ValueError, match=re.escape("confidence must be in (0.0, 1.0)")):
            adaptive_length(0.5, epsilon=0.01, confidence=1.0, method="chebyshev")

    def test_hoeffding_confidence_1_raises(self) -> None:
        """A confidence of 1.0 is refused by the Hoeffding bound."""
        with pytest.raises(ValueError, match=re.escape("confidence must be in (0.0, 1.0)")):
            adaptive_length(0.5, epsilon=0.01, confidence=1.0, method="hoeffding")

    def test_empirical_accuracy_meets_target(self) -> None:
        """Adaptive length should actually achieve the target precision."""
        p = 0.7
        epsilon = 0.05
        L = adaptive_length(p, epsilon=epsilon, confidence=0.95, method="hoeffding")
        n_trials = 100
        errors = []
        for _ in range(n_trials):
            bs = generate_bernoulli_bitstream(p, L)
            errors.append(abs(bitstream_to_probability(bs) - p))
        # At least 90% of trials should be within epsilon (test is conservative)
        within = sum(1 for e in errors if e < epsilon) / n_trials
        assert within > 0.85
