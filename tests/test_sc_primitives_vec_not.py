# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVecNot from former test_sc_primitives.py

"""Focused suite: TestVecNot from former test_sc_primitives.py."""

from __future__ import annotations

from tests.sc_primitives_support import *  # noqa: F403

class TestVecNot:
    def test_complement_probability(self):
        p = 0.7
        packed, length = _bernoulli_packed(p, 10000, seed=42)
        result = vec_not(packed)
        estimated = _prob(result, length)
        np.testing.assert_allclose(estimated, 1.0 - p, atol=0.03)

    def test_double_not_identity(self):
        packed, length = _bernoulli_packed(0.6, 1000, seed=99)
        result = vec_not(vec_not(packed))
        assert np.array_equal(packed, result)
