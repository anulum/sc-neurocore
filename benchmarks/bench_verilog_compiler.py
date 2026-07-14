# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-bound equation-to-Verilog compiler benchmark

"""Measure representative registered and folded Verilog compilation paths.

The benchmark rotates seven deterministic workloads and binds results to every
compiler source file. It records ordinary loaded-host timings only; it does not
claim isolated-core throughput or cross-language performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from sc_neurocore.compiler.verilog_compiler import compile_to_datapath, compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron, from_equations
from sc_neurocore.neurons.universal_dsl import UniversalNeuron


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks/results/bench_verilog_compiler.json"
SOURCE_PATHS = (
    "src/sc_neurocore/compiler/verilog_compiler.py",
    "src/sc_neurocore/compiler/_verilog_integrators.py",
    "src/sc_neurocore/compiler/_verilog_neuron_core.py",
    "src/sc_neurocore/compiler/_verilog_registered_module.py",
    "src/sc_neurocore/compiler/_verilog_folded_datapath.py",
    "src/sc_neurocore/compiler/verilog_compiler_config.py",
    "src/sc_neurocore/compiler/verilog_expr_emitter.py",
    "src/sc_neurocore/compiler/expr_lut_tables.py",
    "src/sc_neurocore/neurons/equation_builder.py",
    "benchmarks/bench_verilog_compiler.py",
)


@dataclass(frozen=True)
class CompilerCase:
    """One named deterministic compiler workload."""

    name: str
    compile: Callable[[], str]


def _positive_int(raw_value: str) -> int:
    """Parse a strictly positive command-line integer."""
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _sha256_bytes(payload: bytes) -> str:
    """Return a hexadecimal SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    """Hash one source file without loading it as text."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    """Bind the evidence payload to every governed Verilog compiler source."""
    hashes: dict[str, str] = {}
    for relative_path in SOURCE_PATHS:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"benchmark source is missing: {relative_path}")
        hashes[relative_path] = _sha256_file(path)
    return hashes


def _read_first_line(path: Path) -> str:
    """Read one optional sysfs value without making metadata a run blocker."""
    try:
        return path.read_text(encoding="utf-8").strip().splitlines()[0]
    except (FileNotFoundError, IndexError, OSError):
        return "unavailable"


def _affinity() -> list[int]:
    """Return the current logical-CPU affinity mask."""
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return []


def _cpu_model() -> str:
    """Return the first Linux CPU model description when present."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _cases() -> tuple[CompilerCase, ...]:
    """Build the fixed workload set through maintained public model surfaces."""
    lif = from_equations(
        "dv/dt = -v/tau + I",
        params={"tau": 10.0},
        init={"v": 0.0},
        dt=0.1,
    )
    rk4 = EquationNeuron(
        equations={"v": "v - w + I", "w": "v + w"},
        parameters={"v_reset": -1.0},
        state={"v": 0.0, "w": 0.0},
        threshold="v >= 1.0",
        reset={"v": "v_reset"},
        dt=0.25,
        method="rk4",
    )
    substep = EquationNeuron(
        equations={
            "v": "v - v * v * v / 3.0 - w + I",
            "w": "epsilon * (v + a - b * w)",
        },
        parameters={"a": 0.7, "b": 0.8, "epsilon": 0.08},
        state={"v": -1.0, "w": -0.5},
        threshold="v >= 1.0",
        detection="crossing",
        dt=0.1,
        method="rk4",
        substeps=4,
    )
    escape = UniversalNeuron.from_schema("escape_rate").to_equation_neuron()
    sqrt_map = EquationNeuron(
        equations={"v": "sqrt(v)"},
        state={"v": 4.0},
        dt=1.0,
        method="map",
    )
    nearest_half_map = EquationNeuron(
        equations={"v": "v * 0.5"},
        state={"v": -1.0 / 256.0},
        dt=1.0,
        method="map",
    )
    return (
        CompilerCase("euler_registered", lambda: compile_to_verilog(lif, module_name="bench_lif")),
        CompilerCase("rk4_registered", lambda: compile_to_verilog(rk4, module_name="bench_rk4")),
        CompilerCase(
            "substep_rk4_registered",
            lambda: compile_to_verilog(substep, module_name="bench_substep"),
        ),
        CompilerCase(
            "escape_rate_registered",
            lambda: compile_to_verilog(
                escape,
                module_name="bench_escape",
                data_width=48,
                fraction=24,
            ),
        ),
        CompilerCase(
            "euler_folded_parameter_port",
            lambda: compile_to_datapath(
                lif,
                module_name="bench_lif_pe",
                param_ports=("tau",),
            ),
        ),
        CompilerCase(
            "sqrt_map_registered",
            lambda: compile_to_verilog(sqrt_map, module_name="bench_sqrt_map"),
        ),
        CompilerCase(
            "nearest_half_map_registered",
            lambda: compile_to_verilog(
                nearest_half_map,
                module_name="bench_nearest_half_map",
                rounding="nearest",
            ),
        ),
    )


def run_benchmark(
    *,
    samples: int,
    warmup: int,
    other_heavy_jobs_running: str,
    other_heavy_jobs_note: str,
    command: str,
) -> dict[str, object]:
    """Run all compiler cases and return a provenance-bearing JSON payload."""
    if samples <= 0 or warmup < 0:
        raise ValueError("samples must be positive and warmup must be non-negative")
    if other_heavy_jobs_running not in {"yes", "no", "unknown"}:
        raise ValueError("other_heavy_jobs_running must be yes, no, or unknown")

    cases = _cases()
    reference: dict[str, tuple[str, int]] = {}
    for case in cases:
        output = case.compile()
        reference[case.name] = (
            _sha256_bytes(output.encode("utf-8")),
            len(output.splitlines()),
        )
        for _ in range(warmup):
            if case.compile() != output:
                raise RuntimeError(f"compiler output is non-deterministic: {case.name}")

    load_before = list(os.getloadavg())
    timings: dict[str, list[int]] = {case.name: [] for case in cases}
    for sample_index in range(samples):
        offset = sample_index % len(cases)
        ordered = (*cases[offset:], *cases[:offset])
        for case in ordered:
            started = time.perf_counter_ns()
            output = case.compile()
            timings[case.name].append(time.perf_counter_ns() - started)
            digest, line_count = reference[case.name]
            if _sha256_bytes(output.encode("utf-8")) != digest:
                raise RuntimeError(f"compiler output changed during sampling: {case.name}")
            if len(output.splitlines()) != line_count:
                raise RuntimeError(f"compiler line count changed during sampling: {case.name}")
    load_after = list(os.getloadavg())

    rows: dict[str, object] = {}
    for case in cases:
        values = timings[case.name]
        digest, line_count = reference[case.name]
        rows[case.name] = {
            "samples_ns": values,
            "median_ms": statistics.median(values) / 1_000_000.0,
            "minimum_ms": min(values) / 1_000_000.0,
            "maximum_ms": max(values) / 1_000_000.0,
            "output_sha256": digest,
            "output_lines": line_count,
        }

    affinity = _affinity()
    cpu = affinity[0] if affinity else 0
    frequency_root = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "evidence_class": "local_regression",
        "command": command,
        "working_directory": "repository_root",
        "workload": {
            "samples_per_case": samples,
            "warmup_per_case": warmup,
            "sampling": "round_robin_rotating_start",
            "timed_region": "compiler_call_only",
        },
        "implementation_scope": {
            "authority": "python_equation_to_verilog_compiler",
            "cross_language_comparison": False,
            "reason": (
                "The removed Go, Julia, Mojo, and Rust safety files were unwired, "
                "non-executable generated stubs rather than compiler implementations."
            ),
        },
        "isolation": {
            "classification": "loaded_host",
            "exclusive_core_reserved": False,
            "process_affinity": affinity,
            "other_heavy_jobs_running": other_heavy_jobs_running,
            "other_heavy_jobs_note": other_heavy_jobs_note,
            "load_average_before": load_before,
            "load_average_after": load_after,
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "scaling_governor": _read_first_line(frequency_root / "scaling_governor"),
            "scaling_current_khz_after": _read_first_line(frequency_root / "scaling_cur_freq"),
        },
        "source_sha256": _source_hashes(),
        "cases": rows,
        "interpretation": (
            "These timings are loaded-host local-regression evidence, not promotion-grade "
            "latency or throughput claims."
        ),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write benchmark evidence atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse benchmark sampling and evidence controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=_positive_int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--other-heavy-jobs-running",
        choices=("yes", "no", "unknown"),
        default="unknown",
    )
    parser.add_argument("--other-heavy-jobs-note", default="not disclosed")
    parsed = parser.parse_args(argv)
    if cast(int, parsed.warmup) < 0:
        parser.error("--warmup must be non-negative")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI and write its JSON evidence artifact."""
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    parsed = _parse_args(arguments)
    command = shlex.join(("python", "benchmarks/bench_verilog_compiler.py", *arguments))
    payload = run_benchmark(
        samples=cast(int, parsed.samples),
        warmup=cast(int, parsed.warmup),
        other_heavy_jobs_running=cast(str, parsed.other_heavy_jobs_running),
        other_heavy_jobs_note=cast(str, parsed.other_heavy_jobs_note),
        command=command,
    )
    output = cast(Path, parsed.json)
    _write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
