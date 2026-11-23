# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStepFromBioRates from former test_arcane_zenith.py

"""Focused suite: TestStepFromBioRates from former test_arcane_zenith.py."""

from __future__ import annotations

import pytest
from sc_neurocore.arcane_zenith import (
    ArcaneZenithCognitiveCore,
    create_arcane_neuron_with_zenith_plasticity,
)

pytest.importorskip("torch")


class TestStepFromBioRates:
    """Stepping the cognitive core from biological firing rates."""

    @pytest.fixture
    def core(self) -> ArcaneZenithCognitiveCore:
        """Return a cognitive core on the torch backend."""
        return create_arcane_neuron_with_zenith_plasticity(backend="torch")

    def test_populated_dict_advances_by_one_step(self, core: ArcaneZenithCognitiveCore) -> None:
        """A rate dictionary advances the clock by one step."""
        core.step_from_bio_rates({0: 10.0, 1: 20.0, 2: 30.0})
        assert core.neuron.get_state()["total_steps"] == 1

    def test_empty_dict_treated_as_zero_current(self, core: ArcaneZenithCognitiveCore) -> None:
        """An empty rate dictionary still advances one step."""
        # Empty dict → mean 0.0 → equivalent to step(0.0). Must not raise.
        core.step_from_bio_rates({})
        assert core.neuron.get_state()["total_steps"] == 1

    def test_multiple_calls_keep_parameters_bounded(self, core: ArcaneZenithCognitiveCore) -> None:
        """Repeated rate steps keep every meta-parameter in range."""
        for i in range(100):
            core.step_from_bio_rates({0: float(i % 50), 1: float((i * 3) % 40)})
        assert 1000.0 <= core.neuron.tau_deep <= 50000.0
        assert 0.01 <= core.neuron.surprise_baseline <= 0.5
        assert 0.0 <= core.neuron.delta_conf <= 1.0
        assert 0.001 <= core.neuron.lr_base <= 0.1
