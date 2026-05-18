# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.security -- Tier: research (experimental /

"""sc_neurocore.security -- Tier: research (experimental / research)."""

__tier__ = "research"

from .ethics import AsimovGovernor, ActionRequest
from .immune import DigitalImmuneSystem
from .side_channel_metrics import (
    ClassActivityProxy,
    SideChannelMetricError,
    SwitchingActivitySummary,
    compute_class_activity_proxy,
    compute_switching_activity,
)
from .side_channel_benchmark import (
    SIDE_CHANNEL_BENCHMARK_SCHEMA_VERSION,
    SIDE_CHANNEL_DEPLOY_MANIFEST_SCHEMA_VERSION,
    SideChannelBenchmarkArm,
    SideChannelBenchmarkError,
    SideChannelBenchmarkRecord,
    SideChannelBenchmarkReport,
    SideChannelDeployManifest,
    run_side_channel_leakage_benchmark,
    write_side_channel_benchmark_report,
)
from .thermal_sc_encoding import (
    ActivityBalancedEncoding,
    ActivityBalancedEncodingBatch,
    ActivityBalancedEncodingSummary,
    ThermalSCEncodingConfig,
    ThermalSCEncodingError,
    encode_activity_balanced_probabilities,
    encode_activity_balanced_probability,
)
from .watermark import WatermarkInjector
from .zkp import ZKPVerifier

__all__ = [
    "ActivityBalancedEncoding",
    "ActivityBalancedEncodingBatch",
    "ActivityBalancedEncodingSummary",
    "AsimovGovernor",
    "ActionRequest",
    "ClassActivityProxy",
    "DigitalImmuneSystem",
    "SIDE_CHANNEL_BENCHMARK_SCHEMA_VERSION",
    "SIDE_CHANNEL_DEPLOY_MANIFEST_SCHEMA_VERSION",
    "SideChannelMetricError",
    "SideChannelBenchmarkArm",
    "SideChannelBenchmarkError",
    "SideChannelBenchmarkRecord",
    "SideChannelBenchmarkReport",
    "SideChannelDeployManifest",
    "SwitchingActivitySummary",
    "ThermalSCEncodingConfig",
    "ThermalSCEncodingError",
    "WatermarkInjector",
    "ZKPVerifier",
    "encode_activity_balanced_probabilities",
    "encode_activity_balanced_probability",
    "compute_class_activity_proxy",
    "compute_switching_activity",
    "run_side_channel_leakage_benchmark",
    "write_side_channel_benchmark_report",
]
