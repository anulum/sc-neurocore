# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Formal verification package (Lean 4 bridge)

"""Expose formal network-property compilation and replay helpers."""

from .counterexample_replay import (
    AntagonisticReplayResult,
    PopulationCoactivationReplayResult,
    PopulationInactivityReplayResult,
    PopulationSilenceReplayResult,
    RateBoundReplayResult,
    RefractoryReplayResult,
    TemporalSeparationReplayResult,
    replay_antagonistic_counterexample,
    replay_population_coactivation_counterexample,
    replay_population_inactivity_counterexample,
    replay_population_silence_counterexample,
    replay_rate_bound_counterexample,
    replay_refractory_counterexample,
    replay_temporal_separation_counterexample,
)
from .lean_bridge import FormalProofEngine
from .network_properties import (
    DenseLIFNetworkSpec,
    NetworkAntagonisticOutputExclusion,
    NetworkOutputTemporalSeparation,
    NetworkPopulationCoactivationCap,
    NetworkPopulationInactivityBound,
    NetworkPopulationSilenceAfterCoactivation,
    NetworkRateBound,
    NetworkRefractoryInvariant,
)
from .property_compiler import (
    compile_dense_lif_fixture_rtl,
    compile_network_antagonistic_exclusion_sva,
    compile_network_population_coactivation_sva,
    compile_network_population_inactivity_sva,
    compile_network_population_silence_sva,
    compile_network_rate_bound_sva,
    compile_network_refractory_sva,
    compile_network_temporal_separation_sva,
)
from .report_schema import (
    FORMAL_NETWORK_REPORT_SCHEMA_VERSION,
    FormalReportValidationError,
    validate_formal_network_report,
)

__all__ = [
    "DenseLIFNetworkSpec",
    "FORMAL_NETWORK_REPORT_SCHEMA_VERSION",
    "FormalReportValidationError",
    "FormalProofEngine",
    "NetworkAntagonisticOutputExclusion",
    "NetworkOutputTemporalSeparation",
    "NetworkPopulationCoactivationCap",
    "NetworkPopulationInactivityBound",
    "NetworkPopulationSilenceAfterCoactivation",
    "NetworkRateBound",
    "NetworkRefractoryInvariant",
    "AntagonisticReplayResult",
    "PopulationCoactivationReplayResult",
    "PopulationInactivityReplayResult",
    "PopulationSilenceReplayResult",
    "RateBoundReplayResult",
    "RefractoryReplayResult",
    "TemporalSeparationReplayResult",
    "compile_dense_lif_fixture_rtl",
    "compile_network_antagonistic_exclusion_sva",
    "compile_network_population_coactivation_sva",
    "compile_network_population_inactivity_sva",
    "compile_network_population_silence_sva",
    "compile_network_rate_bound_sva",
    "compile_network_refractory_sva",
    "compile_network_temporal_separation_sva",
    "replay_antagonistic_counterexample",
    "replay_population_coactivation_counterexample",
    "replay_population_inactivity_counterexample",
    "replay_population_silence_counterexample",
    "replay_rate_bound_counterexample",
    "replay_refractory_counterexample",
    "replay_temporal_separation_counterexample",
    "validate_formal_network_report",
]
