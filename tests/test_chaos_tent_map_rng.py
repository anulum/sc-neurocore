# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTentMapRNG from former test_chaos.py

"""Focused suite: TestTentMapRNG from former test_chaos.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.chaos import (
    TentMapRNG,
)


class TestTentMapRNG:
    """The tent-map chaotic random generator."""

    def test_output_range(self) -> None:
        """Values lie strictly between 0 and 1."""
        rng = TentMapRNG(x=0.37)
        vals = rng.random(10_000)
        assert vals.min() > 0.0
        assert vals.max() < 1.0

    def test_deterministic(self) -> None:
        """The same seed gives the same values."""
        a = TentMapRNG(x=0.37)
        b = TentMapRNG(x=0.37)
        np.testing.assert_array_equal(a.random(100), b.random(100))

    def test_uniform_distribution(self) -> None:
        """Values average about 0.5."""
        rng = TentMapRNG(x=0.37)
        vals = rng.random(100_000)
        assert abs(vals.mean() - 0.5) < 0.02

    def test_bitstream(self) -> None:
        """A bitstream averages about 0.5."""
        rng = TentMapRNG(x=0.37)
        bits = rng.generate_bitstream(0.5, 10_000)
        assert bits.dtype == np.uint8
        assert abs(bits.mean() - 0.5) < 0.05

    def test_reset(self) -> None:
        """A reset reproduces the sequence."""
        rng = TentMapRNG(x=0.37)
        first = rng.random(50)
        rng.reset()
        second = rng.random(50)
        np.testing.assert_array_equal(first, second)

    def test_invalid_mu(self) -> None:
        """A mu outside its range is rejected."""
        with pytest.raises(ValueError, match="mu must be in"):
            TentMapRNG(mu=0.5, x=0.5)

    def test_invalid_x(self) -> None:
        """A seed outside (0, 1) is rejected."""
        with pytest.raises(ValueError, match="x must be in"):
            TentMapRNG(x=0.0)

    def test_state_property(self) -> None:
        """The state lies strictly between 0 and 1."""
        rng = TentMapRNG(x=0.37)
        assert 0.0 < rng.state < 1.0

    def test_collapse_guard(self) -> None:
        """The map never collapses to 0."""
        # x=0.5 at mu=2.0 exactly: step 1 → 1.0, step 2 → 0.0 → guard rescues
        rng = TentMapRNG(mu=2.0, x=0.5)
        vals = rng.random(100)
        assert vals.min() > 0.0
