# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestChaoticRNG from former test_bio_chaos_spatial_learning.py

"""Focused suite: TestChaoticRNG from former test_bio_chaos_spatial_learning.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.chaos.rng import (
    ChaoticRNG,
)


class TestChaoticRNG:
    """The chaotic logistic-map random generator."""

    def test_burn_in_changes_state(self) -> None:
        """Burn-in moves the state from its seed."""
        rng = ChaoticRNG()
        assert rng.x != 0.5

    def test_random_shape_and_range(self) -> None:
        """Random values have the asked shape and lie in [0, 1]."""
        vals = ChaoticRNG().random(200)
        assert vals.shape == (200,)
        assert np.all((vals >= 0) & (vals <= 1))

    def test_deterministic_same_initial(self) -> None:
        """The same seed gives the same values."""
        a = ChaoticRNG(r=4.0, x=0.3)
        b = ChaoticRNG(r=4.0, x=0.3)
        np.testing.assert_array_equal(a.random(50), b.random(50))

    def test_bitstream_shape(self) -> None:
        """A bitstream has the asked length and uint8 dtype."""
        bits = ChaoticRNG().generate_bitstream(0.5, 100)
        assert bits.shape == (100,)
        assert bits.dtype == np.uint8

    def test_bitstream_extremes(self) -> None:
        """Probabilities 0 and 1 give all zeros and all ones."""
        assert np.all(ChaoticRNG().generate_bitstream(0.0, 100) == 0)
        assert np.all(ChaoticRNG().generate_bitstream(1.0, 100) == 1)
