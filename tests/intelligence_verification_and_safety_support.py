# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_intelligence_verification_and_safety.py

from __future__ import annotations

import unittest
from sc_neurocore.compiler.intelligence import (
    configure_approximation,
    explore_pareto,
    generate_dvfs_controller,
    ingest_telemetry,
    model_energy_harvest,
    predict_aging,
    predict_reliability,
    protect_ip_pqc,
    run_fault_campaign,
    verify_timing_closure,
)
from sc_neurocore.compiler.platforms import get_profile

__all__ = [
    "unittest",
    "configure_approximation",
    "explore_pareto",
    "generate_dvfs_controller",
    "ingest_telemetry",
    "model_energy_harvest",
    "predict_aging",
    "predict_reliability",
    "protect_ip_pqc",
    "run_fault_campaign",
    "verify_timing_closure",
    "get_profile",
]
