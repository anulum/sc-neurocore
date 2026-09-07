# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NeuroBench-compatible benchmark framework

"""NeuroBench-compatible benchmarking for SC-NeuroCore models.

The stochastic-backprop benchmark is the only member that needs ``torch``, and
it is resolved lazily so importing this package does not. Everything else here
-- the metrics, the MLPerf-SC schema, runner and report, the online-O1
adaptation benchmark and the task registry -- runs without it, and eagerly
importing the one torch member made an optional extra mandatory for all of
them: `import sc_neurocore.benchmarks` raised ``ModuleNotFoundError: torch`` in
any environment installed without the ``training``, ``research`` or ``full``
extra, which is every environment installed with ``dev`` alone.

The names still resolve exactly as before, so ``from sc_neurocore.benchmarks
import build_stochastic_backprop_benchmark`` works and raises the missing
dependency at that point rather than at package import. The idiom is PEP 562,
the same one the root package uses for the same reason.
"""

import importlib
from typing import Any

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
from .tasks import TASKS


_TORCH_BACKED_SYMBOLS = {
    "STOCHASTIC_BACKPROP_BENCHMARK_SCHEMA_VERSION": "stochastic_backprop",
    "STOCHASTIC_BACKPROP_ESTIMATOR_REGRESSION_SCHEMA_VERSION": "stochastic_backprop",
    "build_stochastic_backprop_benchmark": "stochastic_backprop",
    "build_stochastic_backprop_estimator_regression_manifest": "stochastic_backprop",
    "write_stochastic_backprop_benchmark": "stochastic_backprop",
    "write_stochastic_backprop_estimator_regression_manifest": "stochastic_backprop",
}
"""The members that need ``torch``, and the module each comes from."""


def __getattr__(name: str) -> Any:
    """Resolve a torch-backed benchmark symbol on first use (PEP 562).

    Parameters
    ----------
    name : str
        The attribute being looked up.

    Returns
    -------
    Any
        The resolved symbol.

    Raises
    ------
    AttributeError
        When the name is not part of this package's public surface.
    ModuleNotFoundError
        When the name needs ``torch`` and it is not installed. The failure
        happens here, at the point of use, rather than at package import, so
        the rest of the benchmarking surface stays available without the
        optional extra.
    """
    source = _TORCH_BACKED_SYMBOLS.get(name)
    if source is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{source}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the public surface, including the lazily resolved members.

    Returns
    -------
    list of str
        Every public name, sorted.
    """
    return sorted(set(__all__) | set(globals()))


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
