# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NeuroBench-compatible benchmark framework

"""NeuroBench-compatible benchmarking for SC-NeuroCore models."""

from .metrics import compute_metrics, BenchmarkResult
from .mlperf_sc_schema import (
    MLPERF_SC_RESULT_SCHEMA_VERSION,
    MLPerfSCArtifact,
    MLPerfSCArea,
    MLPerfSCEvidence,
    MLPerfSCExecution,
    MLPerfSCMetrics,
    MLPerfSCResult,
    MLPerfSCRun,
    MLPerfSCValidationError,
    mlperf_sc_result_to_dict,
    validate_mlperf_sc_result,
)
from .mlperf_sc_runner import run_mlperf_sc_fixture
from .mlperf_sc_report import MLPERF_SC_REPORT_SCHEMA_VERSION, aggregate_mlperf_sc_results
from .online_o1_adaptation import (
    ONLINE_O1_ADAPTATION_BENCHMARK_SCHEMA_VERSION,
    build_online_o1_adaptation_benchmark,
    write_online_o1_adaptation_benchmark,
)
from .stochastic_backprop import (
    STOCHASTIC_BACKPROP_BENCHMARK_SCHEMA_VERSION,
    STOCHASTIC_BACKPROP_ESTIMATOR_REGRESSION_SCHEMA_VERSION,
    build_stochastic_backprop_benchmark,
    build_stochastic_backprop_estimator_regression_manifest,
    write_stochastic_backprop_benchmark,
    write_stochastic_backprop_estimator_regression_manifest,
)
from .tasks import TASKS

__all__ = [
    "BenchmarkResult",
    "MLPERF_SC_RESULT_SCHEMA_VERSION",
    "MLPERF_SC_REPORT_SCHEMA_VERSION",
    "MLPerfSCArtifact",
    "MLPerfSCArea",
    "MLPerfSCEvidence",
    "MLPerfSCExecution",
    "MLPerfSCMetrics",
    "MLPerfSCResult",
    "MLPerfSCRun",
    "MLPerfSCValidationError",
    "ONLINE_O1_ADAPTATION_BENCHMARK_SCHEMA_VERSION",
    "STOCHASTIC_BACKPROP_BENCHMARK_SCHEMA_VERSION",
    "STOCHASTIC_BACKPROP_ESTIMATOR_REGRESSION_SCHEMA_VERSION",
    "TASKS",
    "aggregate_mlperf_sc_results",
    "build_online_o1_adaptation_benchmark",
    "build_stochastic_backprop_benchmark",
    "build_stochastic_backprop_estimator_regression_manifest",
    "compute_metrics",
    "mlperf_sc_result_to_dict",
    "run_mlperf_sc_fixture",
    "validate_mlperf_sc_result",
    "write_online_o1_adaptation_benchmark",
    "write_stochastic_backprop_benchmark",
    "write_stochastic_backprop_estimator_regression_manifest",
]
