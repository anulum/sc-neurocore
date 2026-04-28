# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Automatic FPGA resource optimiser

"""Automatically compress and tune SNNs for target hardware."""

from .observation_loader import (
    ObservationLoadError,
    load_observations,
    load_synthesis_observation,
    observation_from_synthesis_reports,
    observations_from_payload,
)
from .resource_optimizer import fit_to_target, OptimizationResult
from .synthesis_evidence import build_payload_from_reports, write_payload
from .surrogate_sc_optimizer import (
    BenchmarkObservation,
    SurrogateOptimizerReport,
    SurrogateSCOptimizer,
    TargetHardwareProfile,
)

__all__ = [
    "fit_to_target",
    "OptimizationResult",
    "ObservationLoadError",
    "load_observations",
    "load_synthesis_observation",
    "observation_from_synthesis_reports",
    "observations_from_payload",
    "build_payload_from_reports",
    "write_payload",
    "BenchmarkObservation",
    "SurrogateOptimizerReport",
    "SurrogateSCOptimizer",
    "TargetHardwareProfile",
]
