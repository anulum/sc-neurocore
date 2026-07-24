# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAstrocyte from former test_model_astrocyte.py

"""Focused suite: TestAstrocyte from former test_model_astrocyte.py."""

from __future__ import annotations

from tests.model_astrocyte_support import *  # noqa: F403


class TestAstrocyte:
    def test_calcium_dynamics(self):
        from sc_neurocore.neurons.models.astrocyte import AstrocyteModel

        n = AstrocyteModel()
        for _ in range(200):
            ca = n.step(1.0)
        assert isinstance(ca, float)
        assert ca > 0.0
