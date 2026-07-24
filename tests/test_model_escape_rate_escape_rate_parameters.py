# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEscapeRateParameters from former test_model_escape_rate.py

"""Focused suite: TestEscapeRateParameters from former test_model_escape_rate.py."""

from __future__ import annotations

from tests.model_escape_rate_support import *  # noqa: F403


class TestEscapeRateParameters:
    def test_tau_m_controls_v_dynamics(self):
        n_fast = EscapeRateNeuron(tau_m=2.0)
        n_slow = EscapeRateNeuron(tau_m=50.0)
        n_fast.step(20.0)
        n_slow.step(20.0)
        assert abs(n_fast.v - (-70.0)) > abs(n_slow.v - (-70.0))

    def test_resistance_scales_input(self):
        n_low = EscapeRateNeuron(resistance=0.5)
        n_high = EscapeRateNeuron(resistance=2.0)
        n_low.step(20.0)
        n_high.step(20.0)
        assert abs(n_high.v - (-70.0)) > abs(n_low.v - (-70.0))
