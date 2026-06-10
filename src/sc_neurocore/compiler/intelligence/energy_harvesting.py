# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Energy harvesting scheduler

"""Energy budget modeling and update scheduling for energy-harvesting edge."""

from __future__ import annotations

from dataclasses import dataclass

from .thermal_analysis import _require_finite_non_negative, _require_finite_positive


@dataclass
class EnergySchedule:
    """Energy-aware neuron update schedule.

    Attributes
    ----------
    total_neurons : int
        Total neurons.
    energy_budget_uj : float
        Energy budget per epoch (µJ).
    neurons_per_epoch : int
        Neurons updatable within budget.
    update_order : list[int]
        Priority-ordered neuron indices.
    epoch_duration_ms : float
        Epoch duration.
    duty_cycle : float
        Fraction of neurons updated per epoch.
    """

    total_neurons: int
    energy_budget_uj: float
    neurons_per_epoch: int
    update_order: list[int]
    epoch_duration_ms: float
    duty_cycle: float


@dataclass
class EnergyHarvestBudget:
    """Energy harvesting feasibility analysis.

    Attributes
    ----------
    harvester_power_uw : float
    design_power_uw : float
    energy_positive : bool
    recommended_duty_cycle : float
    margin_pct : float
    """

    harvester_power_uw: float
    design_power_uw: float
    energy_positive: bool
    recommended_duty_cycle: float
    margin_pct: float


def generate_energy_schedule(
    neuron_count: int,
    *,
    energy_budget_uj: float = 10.0,
    energy_per_neuron_nj: float = 50.0,
    epoch_duration_ms: float = 10.0,
    priority_neurons: list[int] | None = None,
) -> EnergySchedule:
    """Generate energy-budget-aware neuron update schedule."""
    if not isinstance(neuron_count, int) or neuron_count <= 0:
        raise ValueError("neuron_count must be a positive integer")
    _require_finite_non_negative(energy_budget_uj, "energy_budget_uj")
    _require_finite_positive(energy_per_neuron_nj, "energy_per_neuron_nj")
    _require_finite_positive(epoch_duration_ms, "epoch_duration_ms")

    budget_nj = energy_budget_uj * 1000
    max_neurons = int(budget_nj / energy_per_neuron_nj)
    updatable = min(max_neurons, neuron_count)

    # Priority ordering
    if priority_neurons:
        order = []
        seen = set()
        for idx in priority_neurons:
            if not isinstance(idx, int) or idx < 0 or idx >= neuron_count:
                raise ValueError("priority_neurons must contain valid neuron indices")
            if idx in seen:
                continue
            seen.add(idx)
            order.append(idx)
        remaining = [i for i in range(neuron_count) if i not in order]
        order.extend(remaining)
    else:
        order = list(range(neuron_count))

    order = order[:updatable]
    duty = updatable / neuron_count if neuron_count > 0 else 0.0

    return EnergySchedule(
        total_neurons=neuron_count,
        energy_budget_uj=energy_budget_uj,
        neurons_per_epoch=updatable,
        update_order=order,
        epoch_duration_ms=epoch_duration_ms,
        duty_cycle=round(duty, 4),
    )


def model_energy_harvest(
    design_power_uw: float,
    *,
    harvester_type: str = "solar",
    harvester_area_cm2: float = 1.0,
    environment: str = "indoor",
) -> EnergyHarvestBudget:
    """Model whether an energy harvester can sustain the neural workload."""
    # Power density lookup (µW/cm²)
    densities = {
        ("solar", "outdoor"): 10000.0,
        ("solar", "indoor"): 10.0,
        ("solar", "industrial"): 50.0,
        ("piezo", "outdoor"): 200.0,
        ("piezo", "indoor"): 100.0,
        ("piezo", "industrial"): 500.0,
        ("thermal", "outdoor"): 25.0,
        ("thermal", "indoor"): 10.0,
        ("thermal", "industrial"): 60.0,
        ("rf", "outdoor"): 1.0,
        ("rf", "indoor"): 0.5,
        ("rf", "industrial"): 2.0,
    }
    density = densities.get((harvester_type, environment), 10.0)
    harvest_power = density * harvester_area_cm2

    energy_positive = harvest_power >= design_power_uw
    if design_power_uw > 0:
        duty_cycle = min(1.0, harvest_power / design_power_uw)
        margin = ((harvest_power - design_power_uw) / design_power_uw) * 100
    else:
        duty_cycle = 1.0
        margin = 100.0

    return EnergyHarvestBudget(
        harvester_power_uw=round(harvest_power, 2),
        design_power_uw=design_power_uw,
        energy_positive=energy_positive,
        recommended_duty_cycle=round(duty_cycle, 4),
        margin_pct=round(margin, 1),
    )
