from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "hdl" / "formal"))

from timing import (  # noqa: E402
    TimingProofOrchestrator,
    TimingProperty,
    emit_kind2_module,
    emit_nuxmv_module,
)

EXAMPLE_SBY = REPO_ROOT / "hdl" / "formal" / "timing" / "example_dense_layer_core_latency.sby"


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _cgroup_effective_cpuset() -> str | None:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return _read_text(Path("/sys/fs/cgroup/cpuset.cpus.effective"))


def _cpu_governors(affinity: list[int]) -> dict[str, str]:
    governors: dict[str, str] = {}
    for cpu in affinity[:8]:
        value = _read_text(Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"))
        if value is not None:
            governors[str(cpu)] = value
    return governors


def _host_context() -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0))
    cpuset = _cgroup_effective_cpuset()
    return {
        "affinity_cpus": affinity,
        "affinity_cpu_count": len(affinity),
        "cgroup_effective_cpuset": cpuset,
        "load_average": list(os.getloadavg()),
        "cpu_governors_sample": _cpu_governors(affinity),
        "runtime_cpuset_shield_claimed": cpuset == "10-11" or affinity == [10, 11],
        "isolation_requirement": "Run under a runtime cpuset shield; do not compare with unloaded baselines unless the cgroup_effective_cpuset confirms the isolated cores.",
    }


def _properties() -> list[TimingProperty]:
    signals = [
        ("dense_start_to_done", "latency", "start_pulse", "run_done", 6),
        ("dense_start_to_step", "bounded_liveness", "start_pulse", "step_valid", 2),
        ("dense_step_to_done", "deadline", "step_valid", "run_done", 8),
        ("dense_start_to_running", "bounded_liveness", "start_pulse", "running", 1),
    ]
    properties: list[TimingProperty] = []
    for suffix, scale in (("nominal", 1), ("guarded", 2), ("extended", 3), ("stress", 4)):
        for name, kind, trigger, response, bound in signals:
            properties.append(
                TimingProperty(
                    name=f"{name}_{suffix}",
                    kind=kind,  # type: ignore[arg-type]
                    trigger=trigger,
                    response=response,
                    bound_cycles=bound * scale,
                    description="dense-layer timing proof benchmark property",
                )
            )
    return properties


def run() -> dict[str, Any]:
    properties = _properties()

    emit_start = time.perf_counter()
    nuxmv_models = [emit_nuxmv_module(prop) for prop in properties]
    nuxmv_emit_s = time.perf_counter() - emit_start

    emit_start = time.perf_counter()
    kind2_models = [emit_kind2_module(prop) for prop in properties]
    kind2_emit_s = time.perf_counter() - emit_start

    tmp_root = REPO_ROOT / "benchmarks" / "results" / ".tmp_timing_formal"
    proof = TimingProofOrchestrator(EXAMPLE_SBY, temp_root=tmp_root).prove(timeout_s=120)
    shutil.rmtree(tmp_root, ignore_errors=True)

    return {
        "benchmark": "timing_formal_framework",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "language": "Python+SystemVerilog",
        "language_surfaces": ["Python", "SystemVerilog", "nuXmv", "Kind2"],
        "hardware_measurement_claimed": False,
        "property_count": len(properties),
        "nuxmv_emit_count": len(nuxmv_models),
        "kind2_emit_count": len(kind2_models),
        "nuxmv_emit_s": nuxmv_emit_s,
        "kind2_emit_s": kind2_emit_s,
        "nuxmv_available": shutil.which("nuxmv") is not None,
        "kind2_available": shutil.which("kind2") is not None,
        "cvc5_available": shutil.which("cvc5") is not None,
        "sby_available": shutil.which("sby") is not None,
        "sby_exit_code": proof.exit_code,
        "sby_runtime_s": proof.runtime_s,
        "symbiyosys_passed": proof.passed,
        "symbiyosys_unavailable": list(proof.unavailable),
        "comparison": {
            "Python": {
                "operation": "construct TimingProperty objects and orchestrate proof execution",
                "property_count": len(properties),
            },
            "SystemVerilog": {
                "operation": "prove dense-layer timing monitors through SymbiYosys/cvc5",
                "runtime_s": proof.runtime_s,
                "passed": proof.passed,
            },
            "nuXmv": {
                "operation": "emit bounded timing transition models",
                "model_count": len(nuxmv_models),
                "emit_s": nuxmv_emit_s,
                "execution_available": shutil.which("nuxmv") is not None,
            },
            "Kind2": {
                "operation": "emit Lustre bounded timing nodes",
                "model_count": len(kind2_models),
                "emit_s": kind2_emit_s,
                "execution_available": shutil.which("kind2") is not None,
            },
        },
        "host_context": _host_context(),
        "proof_stdout_tail": proof.stdout_tail,
        "proof_stderr_tail": proof.stderr_tail,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the timing formal framework.")
    parser.add_argument("--json", type=Path, required=True, help="Output JSON artefact path")
    args = parser.parse_args()

    result = run()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
