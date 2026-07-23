# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHybridFisherPosnerLIF from former test_fisher_posner.py

"""Focused suite: TestHybridFisherPosnerLIF from former test_fisher_posner.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
from fisher_posner_support import *  # noqa: F403

class TestHybridFisherPosnerLIF:
    def test_resting_state(self, pool_and_neuron: PoolAndNeuron) -> None:
        _, neuron = pool_and_neuron
        assert neuron.Vm == -70.0
        assert neuron.atp_level == 1.0
        assert not neuron.is_spiking

    def test_subthreshold_no_spike(self, pool_and_neuron: PoolAndNeuron) -> None:
        """Small input should not trigger a spike."""
        _, neuron = pool_and_neuron
        vm, spiked = neuron.step(5.0)
        assert not spiked
        assert vm > -70.0  # slight depolarisation

    def test_suprathreshold_spike(self, pool_and_neuron: PoolAndNeuron) -> None:
        """Large input should trigger a spike."""
        _, neuron = pool_and_neuron
        for _ in range(50):
            _, spiked = neuron.step(100.0)
            if spiked:
                break
        assert neuron._total_spikes > 0

    def test_metabolic_failure(self, pool_and_neuron: PoolAndNeuron) -> None:
        """Depleted ATP should prevent spiking (metabolic failure)."""
        _, neuron = pool_and_neuron
        neuron.atp_level = 0.0
        neuron.Vm = -45.0  # above threshold
        _, spiked = neuron.step(100.0)
        assert not spiked
        assert neuron._metabolic_failures > 0

    def test_atp_regeneration(self, pool_and_neuron: PoolAndNeuron) -> None:
        """ATP should regenerate over time via quantum efficiency."""
        _, neuron = pool_and_neuron
        neuron.atp_level = 0.5
        for _ in range(100):
            neuron.step(0.0)
        assert neuron.atp_level > 0.5

    def test_spike_feeds_back_to_pool(self, pool_and_neuron: PoolAndNeuron) -> None:
        """Spiking should call apply_measurement on the spin pool."""
        pool, neuron = pool_and_neuron
        initial_count = pool._measurement_count
        # Drive to spike
        for _ in range(100):
            _, spiked = neuron.step(100.0)
            if spiked:
                break
        if neuron._total_spikes > 0:
            assert pool._measurement_count > initial_count

    def test_v_property(self, pool_and_neuron: PoolAndNeuron) -> None:
        """v property should alias Vm."""
        _, neuron = pool_and_neuron
        neuron.Vm = -55.0
        assert neuron.v == -55.0
        neuron.v = -60.0
        assert neuron.Vm == -60.0

    def test_reset(self, pool_and_neuron: PoolAndNeuron) -> None:
        _, neuron = pool_and_neuron
        neuron.step(100.0)
        neuron.reset_state()
        assert neuron.Vm == -70.0
        assert neuron.atp_level == 1.0
        assert neuron._total_spikes == 0

    def test_get_state(self, pool_and_neuron: PoolAndNeuron) -> None:
        _, neuron = pool_and_neuron
        state = neuron.get_state()
        assert "neuron_id" in state
        assert "Vm" in state
        assert "atp_level" in state

    def test_invalid_neuron_id(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        with pytest.raises(ValueError):
            HybridFisherPosnerLIF(neuron_id=5, spin_pool=pool)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError):
            HybridFisherPosnerLIF(neuron_id=0, spin_pool="not a pool")  # type: ignore[arg-type]
