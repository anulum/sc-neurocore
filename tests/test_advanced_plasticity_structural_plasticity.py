# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStructuralPlasticity from former test_advanced_plasticity.py

"""Focused suite: TestStructuralPlasticity from former test_advanced_plasticity.py."""

from __future__ import annotations

from tests.advanced_plasticity_support import *  # noqa: F403

class TestStructuralPlasticity:
    def test_prune(self, simple_net):
        _, _, _, proj = simple_net
        proj.data[:] = 0.001
        sp = StructuralPlasticity(prune_threshold=0.05)
        sp.update(proj)
        assert np.sum(proj.data == 0.0) > 0

    def test_grow(self, simple_net):
        _, _, _, proj = simple_net
        proj.data[:] = 0.0
        sp = StructuralPlasticity(growth_rate=1.0, prune_threshold=0.001)
        sp.update(proj)
        assert np.any(proj.data > 0)
