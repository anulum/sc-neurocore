# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAstrocyteIP3Dynamics from former test_model_astrocyte.py

"""Focused suite: TestAstrocyteIP3Dynamics from former test_model_astrocyte.py."""

from __future__ import annotations

from tests.model_astrocyte_support import *  # noqa: F403


class TestAstrocyteIP3Dynamics:
    def test_ip3_increases_with_input(self):
        n = AstrocyteModel()
        for _ in range(1000):
            n.step(1.0)
        assert n.ip3 > 0.5  # initial was 0.5, input adds more

    def test_ip3_decays_without_input(self):
        n = AstrocyteModel()
        n.ip3 = 5.0
        for _ in range(10000):
            n.step(0.0)
        assert n.ip3 < 5.0

    def test_ip3_non_negative(self):
        n = AstrocyteModel()
        for _ in range(50000):
            n.step(0.0)
        assert n.ip3 >= 0.0
