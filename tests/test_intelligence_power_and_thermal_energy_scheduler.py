# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEnergyScheduler from former test_intelligence_power_and_thermal.py

"""Focused suite: TestEnergyScheduler from former test_intelligence_power_and_thermal.py."""

from __future__ import annotations

from typing import Any
import pytest


class TestEnergyScheduler:
    """Energy-aware neuron scheduling."""

    def test_basic_schedule(self) -> None:
        """A schedule covers the neurons with a positive duty cycle."""
        from sc_neurocore.compiler.intelligence import (
            generate_energy_schedule,
        )

        s = generate_energy_schedule(1000)
        assert s.total_neurons == 1000
        assert s.neurons_per_epoch <= 1000
        assert s.duty_cycle > 0

    def test_energy_limited(self) -> None:
        """A small budget limits the neurons per epoch."""
        from sc_neurocore.compiler.intelligence import (
            generate_energy_schedule,
        )

        s = generate_energy_schedule(
            1000,
            energy_budget_uj=1.0,
            energy_per_neuron_nj=100.0,
        )
        assert s.neurons_per_epoch == 10
        assert s.duty_cycle == 0.01

    def test_priority_neurons(self) -> None:
        """Priority neurons are updated first."""
        from sc_neurocore.compiler.intelligence import (
            generate_energy_schedule,
        )

        s = generate_energy_schedule(
            100,
            priority_neurons=[50, 51, 52],
        )
        assert s.update_order[0] == 50
        assert s.update_order[1] == 51

    def test_excess_budget(self) -> None:
        """A large budget runs at full duty."""
        from sc_neurocore.compiler.intelligence import (
            generate_energy_schedule,
        )

        s = generate_energy_schedule(
            10,
            energy_budget_uj=1000.0,
        )
        assert s.neurons_per_epoch == 10
        assert s.duty_cycle == 1.0

    def test_rejects_invalid_schedule_inputs(self) -> None:
        """Each invalid schedule input is rejected."""
        from sc_neurocore.compiler.intelligence import (
            generate_energy_schedule,
        )

        invalid_cases: list[tuple[dict[str, Any], str]] = [
            ({"neuron_count": 0}, "neuron_count"),
            ({"neuron_count": 10, "energy_budget_uj": -1.0}, "energy_budget_uj"),
            ({"neuron_count": 10, "energy_per_neuron_nj": 0.0}, "energy_per_neuron_nj"),
            ({"neuron_count": 10, "epoch_duration_ms": 0.0}, "epoch_duration_ms"),
            ({"neuron_count": 10, "priority_neurons": [-1]}, "priority_neurons"),
            ({"neuron_count": 10, "priority_neurons": [10]}, "priority_neurons"),
        ]
        for kwargs, message in invalid_cases:
            with pytest.raises(ValueError, match=message):
                generate_energy_schedule(**kwargs)

    def test_priority_neurons_are_deduplicated(self) -> None:
        """Repeated priority neurons are listed once."""
        from sc_neurocore.compiler.intelligence import (
            generate_energy_schedule,
        )

        s = generate_energy_schedule(5, priority_neurons=[2, 2, 1])
        assert s.update_order[:2] == [2, 1]
        assert len(s.update_order) == len(set(s.update_order))
