# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_security_side_channel_benchmark.py

from __future__ import annotations

"""Support extracted from test_security_side_channel_benchmark.py."""

import json


import pytest


from sc_neurocore.security import side_channel_benchmark as benchmark_mod


from sc_neurocore.security.side_channel_benchmark import (
    SIDE_CHANNEL_BENCHMARK_SCHEMA_VERSION,
    SIDE_CHANNEL_DEPLOY_MANIFEST_SCHEMA_VERSION,
    SideChannelBenchmarkArm,
    SideChannelBenchmarkError,
    SideChannelBenchmarkRecord,
    SideChannelDeployManifest,
    SideChannelBenchmarkReport,
    _correlated_activity_fixture_stream,
    _arm_payload,
    _class_proxy_payload,
    _deploy_manifest_payload,
    _report_payload,
    _with_artifact_path,
    run_side_channel_leakage_benchmark,
    write_side_channel_benchmark_report,
)


from sc_neurocore.security.side_channel_metrics import (
    SideChannelMetricError,
    compute_class_activity_proxy,
    compute_switching_activity,
)


from sc_neurocore.security.thermal_sc_encoding import ThermalSCEncodingConfig


__all__ = [
    "json",
    "pytest",
    "benchmark_mod",
    "SIDE_CHANNEL_BENCHMARK_SCHEMA_VERSION",
    "SIDE_CHANNEL_DEPLOY_MANIFEST_SCHEMA_VERSION",
    "SideChannelBenchmarkArm",
    "SideChannelBenchmarkError",
    "SideChannelBenchmarkRecord",
    "SideChannelDeployManifest",
    "SideChannelBenchmarkReport",
    "_correlated_activity_fixture_stream",
    "_arm_payload",
    "_class_proxy_payload",
    "_deploy_manifest_payload",
    "_report_payload",
    "_with_artifact_path",
    "run_side_channel_leakage_benchmark",
    "write_side_channel_benchmark_report",
    "SideChannelMetricError",
    "compute_class_activity_proxy",
    "compute_switching_activity",
    "ThermalSCEncodingConfig",
]
