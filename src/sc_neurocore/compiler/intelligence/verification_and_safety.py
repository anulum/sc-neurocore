# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verification and safety facade

"""Verification, safety certification, and reliability prediction facade."""

from __future__ import annotations

from .aging_reliability import (
    AgingPrediction,
    ReliabilityEstimate,
    predict_aging,
    predict_reliability,
)
from .bit_true_kernel import (
    generate_bittrue_kernel,
)
from .cdc_analyzer import (
    CDCReport,
    analyze_cdc,
)
from .compliance_matrix import (
    ComplianceEntry,
    format_compliance_report,
    generate_compliance_matrix,
)
from .equivalence_sketch import (
    EquivalenceSketch,
    generate_equivalence_sketch,
)
from .fault_injection import (
    FaultCampaignResult,
    run_fault_campaign,
)
from .fault_tree import (
    FaultTree,
    generate_fault_tree,
)
from .ode_stability import (
    StabilityResult,
    verify_ode_stability,
)
from .regression_watchdog import (
    RegressionCheck,
    check_regression,
)
from .testbench_gen import (
    generate_testbench,
)
from .timing_closure import (
    TimingReport,
    verify_timing_closure,
)

__all__ = [
    "AgingPrediction",
    "CDCReport",
    "ComplianceEntry",
    "EquivalenceSketch",
    "FaultCampaignResult",
    "FaultTree",
    "RegressionCheck",
    "ReliabilityEstimate",
    "StabilityResult",
    "TimingReport",
    "analyze_cdc",
    "check_regression",
    "format_compliance_report",
    "generate_bittrue_kernel",
    "generate_compliance_matrix",
    "generate_equivalence_sketch",
    "generate_fault_tree",
    "generate_testbench",
    "predict_aging",
    "predict_reliability",
    "run_fault_campaign",
    "verify_ode_stability",
    "verify_timing_closure",
]
