# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHybridFisherPosnerLIF from former test_quantum_cognition.py

"""Focused suite: TestHybridFisherPosnerLIF from former test_quantum_cognition.py."""

from __future__ import annotations

from typing import Any, cast
import pytest
from sc_neurocore.quantum_cognition.fisher_posner import (
    HybridFisherPosnerLIF,
)
from sc_neurocore.quantum_cognition.spin_pool import (
    SpinPoolMPS,
)


class TestHybridFisherPosnerLIF:
    """Tests for the quantum-metabolic LIF neuron."""

    @pytest.fixture
    def pool(self) -> SpinPoolMPS:
        """Provide an eight-site spin pool for each neuron."""
        return SpinPoolMPS(n_sites=8)

    def test_init(self, pool: SpinPoolMPS) -> None:
        """A new neuron rests at -70 mV with full ATP."""
        n = HybridFisherPosnerLIF(0, pool)
        assert n.Vm == -70.0
        assert n.atp_level == 1.0
        assert n.id == 0

    def test_init_validation(self, pool: SpinPoolMPS) -> None:
        """A bad neuron id or spin pool is rejected."""
        with pytest.raises(ValueError, match="neuron_id"):
            HybridFisherPosnerLIF(-1, pool)
        with pytest.raises(ValueError, match="exceeds"):
            HybridFisherPosnerLIF(99, pool)
        with pytest.raises(TypeError, match="SpinPoolMPS"):
            HybridFisherPosnerLIF(0, cast(Any, "not_a_pool"))

    def test_subthreshold_no_spike(self, pool: SpinPoolMPS) -> None:
        """A weak input stays below threshold without a spike."""
        n = HybridFisherPosnerLIF(0, pool)
        vm, spiked = n.step(0.0)
        assert not spiked
        assert vm < n.v_threshold

    def test_suprathreshold_spike(self, pool: SpinPoolMPS) -> None:
        """A strong input makes the neuron spike."""
        n = HybridFisherPosnerLIF(0, pool)
        # Large current should cause spike
        for _ in range(100):
            vm, spiked = n.step(50.0)
            if spiked:
                break
        assert n._total_spikes > 0

    def test_metabolic_failure(self, pool: SpinPoolMPS) -> None:
        """With almost no ATP the neuron cannot spike and records a metabolic failure."""
        n = HybridFisherPosnerLIF(0, pool, atp_consumption=0.5)
        n.atp_level = 0.01  # Nearly depleted
        n.Vm = n.v_threshold + 5.0  # Above threshold
        vm, spiked = n.step(0.0)
        # Should fail to spike due to insufficient ATP
        assert not spiked
        assert n._metabolic_failures > 0

    def test_atp_regeneration(self, pool: SpinPoolMPS) -> None:
        """A subthreshold step does not drain a partial ATP level."""
        n = HybridFisherPosnerLIF(0, pool)
        n.atp_level = 0.5  # Partially depleted
        initial_atp = n.atp_level
        n.step(0.0)  # Subthreshold step should regenerate some ATP
        assert n.atp_level >= initial_atp

    def test_spike_feedback_to_spin_pool(self, pool: SpinPoolMPS) -> None:
        """A spike feeds a measurement back to the spin pool."""
        n = HybridFisherPosnerLIF(0, pool)
        initial_count = pool._measurement_count
        # Drive neuron to spike
        for _ in range(200):
            n.step(50.0)
        # Spikes should have triggered measurements
        if n._total_spikes > 0:
            assert pool._measurement_count > initial_count

    def test_get_state(self, pool: SpinPoolMPS) -> None:
        """The state dictionary carries the id, potential and ATP."""
        n = HybridFisherPosnerLIF(3, pool)
        n.step(10.0)
        state = n.get_state()
        assert state["neuron_id"] == 3
        assert "Vm" in state
        assert "atp_level" in state
        assert state["total_steps"] == 1

    def test_reset_state(self, pool: SpinPoolMPS) -> None:
        """A reset restores the rest potential, ATP and spike count."""
        n = HybridFisherPosnerLIF(0, pool)
        for _ in range(50):
            n.step(50.0)
        n.reset_state()
        assert n.Vm == n.v_rest
        assert n.atp_level == 1.0
        assert n._total_spikes == 0

    def test_repr(self, pool: SpinPoolMPS) -> None:
        """The representation names the neuron and its id."""
        n = HybridFisherPosnerLIF(2, pool)
        r = repr(n)
        assert "HybridFisherPosnerLIF" in r
        assert "id=2" in r
