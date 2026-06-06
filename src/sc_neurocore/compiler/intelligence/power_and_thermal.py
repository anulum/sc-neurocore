# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Power and thermal facade

"""Power estimation and thermal-aware compilation facade."""

from __future__ import annotations

from .approximate_computing import (
    ApproximationConfig,
    configure_approximation,
)
from .dvfs_controller import (
    generate_dvfs_controller,
)
from .energy_harvesting import (
    EnergyHarvestBudget,
    EnergySchedule,
    generate_energy_schedule,
    model_energy_harvest,
)
from .power_intent import (
    generate_power_intent,
)
from .power_state_machine import (
    generate_power_state_machine,
)
from .thermal_analysis import (
    ThermalEnvelopeEstimate,
    ThermalEstimate,
    estimate_thermal_envelope,
    generate_thermal_constraints,
    thermal_analysis,
)

__all__ = [
    "ApproximationConfig",
    "EnergyHarvestBudget",
    "EnergySchedule",
    "ThermalEnvelopeEstimate",
    "ThermalEstimate",
    "configure_approximation",
    "estimate_thermal_envelope",
    "generate_dvfs_controller",
    "generate_energy_schedule",
    "generate_power_intent",
    "generate_power_state_machine",
    "generate_thermal_constraints",
    "model_energy_harvest",
    "thermal_analysis",
]
