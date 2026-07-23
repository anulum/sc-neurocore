# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSTPParams from former test_meta_plasticity.py

"""Focused suite: TestSTPParams from former test_meta_plasticity.py."""

from __future__ import annotations

from meta_plasticity_support import *  # noqa: F403

class TestSTPParams:
    def test_to_vector(self):
        p = STPParams()
        assert len(p.to_vector()) == 3

    def test_from_vector_clamps_u(self):
        v = np.array([2.0, 100.0, 50.0])
        p = STPParams.from_vector(v)
        assert p.u_base <= 0.99
