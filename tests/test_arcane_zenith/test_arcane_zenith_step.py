# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStep from former test_arcane_zenith.py

"""Focused suite: TestStep from former test_arcane_zenith.py."""

from __future__ import annotations

from tests.test_arcane_zenith.arcane_zenith_support import *  # noqa: F403

class TestStep:
    @pytest.fixture
    def core(self) -> ArcaneZenithCognitiveCore:
        return create_arcane_neuron_with_zenith_plasticity(backend="torch")

    def test_step_returns_spike_bit(self, core):
        out = core.step(5.0)
        assert out in (0, 1)

    def test_step_advances_neuron_clock(self, core):
        core.step(0.0)
        core.step(0.0)
        core.step(0.0)
        assert core.neuron.get_state()["total_steps"] == 3

    def test_step_keeps_tau_deep_in_biological_range(self, core):
        for _ in range(200):
            core.step(2.5)
        assert 1000.0 <= core.neuron.tau_deep <= 50000.0

    def test_step_keeps_surprise_baseline_in_biological_range(self, core):
        for _ in range(200):
            core.step(2.5)
        assert 0.01 <= core.neuron.surprise_baseline <= 0.5

    def test_step_keeps_delta_conf_in_biological_range(self, core):
        for _ in range(200):
            core.step(2.5)
        assert 0.0 <= core.neuron.delta_conf <= 1.0

    def test_step_keeps_lr_base_in_biological_range(self, core):
        for _ in range(200):
            core.step(2.5)
        assert 0.001 <= core.neuron.lr_base <= 0.1

    def test_step_zero_current_runs_without_error(self, core):
        # No input → neuron stays sub-threshold, but the plasticity rules
        # still step and the meta-parameters still track to biological
        # ranges deterministically.
        for _ in range(50):
            core.step(0.0)
        assert 1000.0 <= core.neuron.tau_deep <= 50000.0
