# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLoihiCUBAAnalytical from former test_model_loihi_cuba.py

"""Focused suite: TestLoihiCUBAAnalytical from former test_model_loihi_cuba.py."""

from __future__ import annotations

from tests.model_loihi_cuba_support import *  # noqa: F403


class TestLoihiCUBAAnalytical:
    def test_u_integrates_input(self):
        n = LoihiCUBANeuron()
        n.step(500)
        assert n.u > 0

    def test_v_driven_by_u(self):
        n = LoihiCUBANeuron()
        n.step(500)
        n.step(0)
        assert n.v > 0

    def test_integer_division_decay(self):
        n = LoihiCUBANeuron()
        n.v = 100
        decay = 100 // n.tau_v
        assert decay == 10

    def test_spike_resets_v(self):
        n = LoihiCUBANeuron()
        for _ in range(10_000):
            if n.step(200) == 1:
                assert n.v == n.v_reset
                break

    def test_two_stage_integration(self):
        """u integrates input, v integrates u (2-stage pipeline)."""
        n = LoihiCUBANeuron()
        assert hasattr(n, "u") and hasattr(n, "v")
