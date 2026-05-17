# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Formal verification package (Lean 4 bridge)

from .counterexample_replay import (
    RateBoundReplayResult,
    RefractoryReplayResult,
    replay_rate_bound_counterexample,
    replay_refractory_counterexample,
)
from .lean_bridge import FormalProofEngine
from .network_properties import DenseLIFNetworkSpec, NetworkRateBound, NetworkRefractoryInvariant
from .property_compiler import (
    compile_dense_lif_fixture_rtl,
    compile_network_rate_bound_sva,
    compile_network_refractory_sva,
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
    "NetworkRateBound",
    "NetworkRefractoryInvariant",
    "RateBoundReplayResult",
    "RefractoryReplayResult",
    "compile_dense_lif_fixture_rtl",
    "compile_network_rate_bound_sva",
    "compile_network_refractory_sva",
    "replay_rate_bound_counterexample",
    "replay_refractory_counterexample",
    "validate_formal_network_report",
]
