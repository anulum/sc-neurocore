# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCuriositySignal from former test_meta_plasticity.py

"""Focused suite: TestCuriositySignal from former test_meta_plasticity.py."""

from __future__ import annotations

from meta_plasticity_support import *  # noqa: F403

class TestCuriositySignal:
    def test_first_update_high_curiosity(self):
        cs = CuriositySignal()
        c = cs.update(np.array([1.0, 2.0, 3.0]))
        assert c == 1.0

    def test_stable_input_low_curiosity(self):
        cs = CuriositySignal(alpha=0.5)
        state = np.array([1.0, 2.0, 3.0])
        cs.update(state)
        for _ in range(20):
            cs.update(state)
        assert cs.curiosity < 0.01

    def test_changing_input_high_curiosity(self):
        cs = CuriositySignal(alpha=0.1)
        cs.update(np.zeros(5))
        c = cs.update(np.ones(5) * 100)
        assert c > 0.5
