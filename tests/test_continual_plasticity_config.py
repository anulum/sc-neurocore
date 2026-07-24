# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlasticityConfig from former test_continual.py

"""Focused suite: TestPlasticityConfig from former test_continual.py."""

from __future__ import annotations

from tests.continual_support import *  # noqa: F403


class TestPlasticityConfig:
    def test_defaults(self):
        c = PlasticityConfig(layer_name="h")
        assert c.rule == "stdp"
        assert c.tau_pre == 20.0
        assert c.w_min == 0.0

    def test_custom_values(self):
        c = PlasticityConfig(layer_name="x", rule="r_stdp", tau_pre=10.0, w_max=2.0)
        assert c.rule == "r_stdp"
        assert c.tau_pre == 10.0
        assert c.w_max == 2.0
