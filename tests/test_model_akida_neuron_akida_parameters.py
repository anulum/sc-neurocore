# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAkidaParameters from former test_model_akida_neuron.py

"""Focused suite: TestAkidaParameters from former test_model_akida_neuron.py."""

from __future__ import annotations

from tests.model_akida_neuron_support import *  # noqa: F403


class TestAkidaParameters:
    @pytest.mark.parametrize("threshold", [50, 100, 200])
    def test_threshold_sweep(self, threshold: int):
        n = AkidaNeuron(threshold=threshold)
        for _ in range(100):
            n.step(50)
        # Should have spiked if threshold low enough
        if threshold <= 50:
            assert n._spiked is True

    @pytest.mark.parametrize("modulation", [0.5, 0.75, 0.9])
    def test_modulation_sweep(self, modulation: float):
        n = AkidaNeuron(modulation=modulation)
        for _ in range(20):
            n.step(50)
        assert isinstance(n.v, int)

    def test_higher_modulation_more_accumulation(self):
        """Higher modulation → slower decay → more total integration."""
        n_low = AkidaNeuron(modulation=0.5, threshold=10000)
        n_high = AkidaNeuron(modulation=0.9, threshold=10000)
        for _ in range(20):
            n_low.step(50)
            n_high.step(50)
        assert n_high.v >= n_low.v
