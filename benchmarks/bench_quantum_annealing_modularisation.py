# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — reproducible quantum-annealing modularisation benchmark

"""Compare bridge import, compilation, and Python solving across source trees."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

from _benchmark_context import load_average, measurement_context

_SCHEMA_VERSION = "sc-neurocore.quantum-annealing-modularisation-benchmark.v1"
_CHILD_PROGRAM = """
import json
import math
import resource
import statistics
import sys
import time

started = time.perf_counter_ns()
import numpy as np
from sc_neurocore.bridges.quantum_annealing import SCToIsing, SimulatedAnnealer
import_ns = time.perf_counter_ns() - started

adjacency = np.zeros((8, 8), dtype=np.float64)
for index in range(7):
    adjacency[index, index + 1] = 1.0
    adjacency[index + 1, index] = 1.0

compile_samples = []
for _ in range(10):
    started = time.perf_counter_ns()
    model = SCToIsing(coupling_scale=1.5, field_scale=0.25).compile(
        adjacency,
        node_labels=[f"n{index}" for index in range(8)],
        biases=np.linspace(-0.2, 0.2, 8),
        name="benchmark_chain",
    )
    compile_samples.append(time.perf_counter_ns() - started)
compile_ns = int(statistics.median(compile_samples))

solve_samples = []
best_energy = 0.0
for repeat in range(5):
    solver = SimulatedAnnealer(n_sweeps=50, seed=42 + repeat)
    started = time.perf_counter_ns()
    result = solver.solve_ising(model, num_reads=2)
    solve_samples.append(time.perf_counter_ns() - started)
    best_energy = float(result["best_energy"])
solve_ns = int(statistics.median(solve_samples))

if model.n_qubits != 8 or len(model.J) != 7 or not math.isfinite(best_energy):
    raise RuntimeError("quantum-annealing probe produced an invalid model or result")
rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
if sys.platform == "darwin":
    rss //= 1024
print(json.dumps({
    "import_ns": import_ns,
    "compile_ns": compile_ns,
    "solve_ns": solve_ns,
    "max_rss_kib": rss,
    "n_qubits": model.n_qubits,
    "n_couplers": len(model.J),
}))
""".strip()
_METRICS = ("subprocess_wall_ns", "import_ns", "compile_ns", "solve_ns", "max_rss_kib")


@dataclass(frozen=True)
class Variant:
    """One source checkout measured by the cold-process probe."""

    label: str
    root: Path


def _parser() -> argparse.ArgumentParser:
    """Build the benchmark command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline-label", default="parent")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--no-affinity", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid counts, labels, and source roots."""
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if args.baseline_label == args.candidate_label:
        raise ValueError("baseline and candidate labels must differ")
    for label, root in (
        (args.baseline_label, args.baseline_root),
        (args.candidate_label, args.candidate_root),
    ):
        if not (root / "src" / "sc_neurocore" / "bridges" / "quantum_annealing.py").is_file():
            raise ValueError(f"{label} root has no quantum-annealing bridge")


def _source_files(root: Path) -> list[Path]:
    """Return the complete facade and responsibility-module source surface."""
    bridges = root / "src" / "sc_neurocore" / "bridges"
    files = [bridges / "quantum_annealing.py", *sorted(bridges.glob("annealing_*.py"))]
    return [path for path in files if path.is_file()]


def _source_digest(root: Path) -> tuple[str, int]:
    """Hash relative names and bytes for the imported bridge surface."""
    files = _source_files(root)
    if not files:
        raise ValueError(f"no quantum-annealing source found below {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), len(files)


def _git_metadata(variant: Variant) -> dict[str, object]:
    """Return path-free Git and source-integrity metadata."""
    head = subprocess.run(
        ["git", "-C", str(variant.root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        [
            "git",
            "-C",
            str(variant.root),
            "status",
            "--porcelain",
            "--",
            "src/sc_neurocore/bridges/quantum_annealing.py",
            ":(glob)src/sc_neurocore/bridges/annealing_*.py",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    git_head = head.stdout.strip() if head.returncode == 0 else None
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    digest, file_count = _source_digest(variant.root)
    reference = (
        f"git:{git_head}" if git_head is not None and not dirty else f"working-tree:{git_head}"
    )
    return {
        "label": variant.label,
        "source_root": reference,
        "local_path_recorded": False,
        "git_head": git_head,
        "quantum_annealing_source_sha256": digest,
        "quantum_annealing_source_file_count": file_count,
        "quantum_annealing_surface_dirty": dirty,
    }


def _select_affinity(args: argparse.Namespace) -> tuple[int | None, str | None, str]:
    """Choose an allowed CPU and optional taskset executable."""
    if args.no_affinity:
        return None, None, "disabled by --no-affinity"
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    requested = args.cpu if args.cpu is not None else (allowed[0] if allowed else None)
    if requested is not None and allowed and requested not in allowed:
        raise ValueError(f"cpu {requested} is outside the process affinity set")
    taskset = shutil.which("taskset")
    if requested is None or taskset is None:
        return None, None, "taskset or an allowed CPU was unavailable"
    return requested, taskset, "taskset affinity pinned; CPU not exclusively isolated"


def _sample(variant: Variant, *, cpu: int | None, taskset: str | None) -> dict[str, int]:
    """Run and validate one cold-process probe."""
    command = [sys.executable, "-c", _CHILD_PROGRAM]
    if cpu is not None and taskset is not None:
        command = [taskset, "--cpu-list", str(cpu), *command]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(variant.root / "src"),
        }
    )
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command,
        cwd=variant.root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    wall_ns = time.perf_counter_ns() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{variant.label} probe failed with {completed.returncode}: {completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{variant.label} probe emitted invalid JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("n_qubits") != 8
        or payload.get("n_couplers") != 7
    ):
        raise RuntimeError(f"{variant.label} probe emitted an invalid contract payload")
    metrics = {"subprocess_wall_ns": wall_ns}
    for metric in _METRICS[1:]:
        value = payload.get(metric)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RuntimeError(f"{variant.label} probe emitted invalid {metric}")
        metrics[metric] = value
    return metrics


def _statistics(samples: list[dict[str, int]]) -> dict[str, object]:
    """Summarize metrics while retaining every raw sample."""
    summary: dict[str, object] = {"samples": samples, "sample_count": len(samples)}
    for metric in _METRICS:
        values = [sample[metric] for sample in samples]
        summary[metric] = {
            "min": min(values),
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "max": max(values),
        }
    return summary


def _median(summary: dict[str, object], metric: str) -> float:
    """Read one numeric median from an internal summary."""
    metric_summary = summary.get(metric)
    if not isinstance(metric_summary, dict):
        raise RuntimeError(f"missing metric summary: {metric}")
    median = metric_summary.get("median")
    if not isinstance(median, (int, float)) or isinstance(median, bool):
        raise RuntimeError(f"missing metric median: {metric}")
    return float(median)


def _comparison(baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    """Compute candidate deltas relative to the parent checkout."""
    comparison: dict[str, object] = {}
    for metric in _METRICS:
        baseline_median = _median(baseline, metric)
        candidate_median = _median(candidate, metric)
        comparison[metric] = {
            "baseline_median": baseline_median,
            "candidate_median": candidate_median,
            "candidate_minus_baseline": candidate_median - baseline_median,
            "candidate_delta_percent": (
                (candidate_median / baseline_median - 1.0) * 100.0
                if baseline_median != 0.0
                else None
            ),
        }
    return comparison


def run(args: argparse.Namespace) -> dict[str, object]:
    """Execute an interleaved parent/candidate benchmark."""
    _validate_args(args)
    variants = (
        Variant(str(args.baseline_label), Path(args.baseline_root).resolve()),
        Variant(str(args.candidate_label), Path(args.candidate_root).resolve()),
    )
    cpu, taskset, affinity_mode = _select_affinity(args)
    load_before = load_average()
    for warmup in range(args.warmups):
        ordered = variants if warmup % 2 == 0 else tuple(reversed(variants))
        for variant in ordered:
            _sample(variant, cpu=cpu, taskset=taskset)

    samples: dict[str, list[dict[str, int]]] = {variant.label: [] for variant in variants}
    for iteration in range(args.iterations):
        ordered = variants if iteration % 2 == 0 else tuple(reversed(variants))
        for variant in ordered:
            samples[variant.label].append(_sample(variant, cpu=cpu, taskset=taskset))
    results = {variant.label: _statistics(samples[variant.label]) for variant in variants}
    return {
        "schema_version": _SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "local_source_refactor_regression",
        "hardware_measurement_claimed": False,
        "operation": "cold import, deterministic 8-node SC-to-Ising compilation, and Python fallback solve",
        "configuration": {
            "iterations": args.iterations,
            "warmups": args.warmups,
            "compile_operations_per_sample": 10,
            "solve_operations_per_sample": 5,
            "n_sweeps": 50,
            "num_reads": 2,
            "interleaving": "variant order reverses on alternating rounds",
            "python_executable": Path(sys.executable).name,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": cpu,
            "affinity_mode": affinity_mode,
            "child_environment": {
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            },
        },
        "commands": {
            "harness": (
                "python benchmarks/bench_quantum_annealing_modularisation.py "
                "--baseline-root <parent-root> --candidate-root <candidate-root> "
                "--output <result.json>"
            ),
            "child": "[taskset --cpu-list <cpu>] <python> -c <cold-start probe>",
        },
        "variants": [_git_metadata(variant) for variant in variants],
        "results": results,
        "comparison": _comparison(results[variants[0].label], results[variants[1].label]),
        "polyglot_applicability": {
            "python": "measured",
            "rust": "maintained_native_kernel_benchmarked_separately",
            "julia": "removed_non_parsing_generated_mirror",
            "mojo": "removed_non_parsing_generated_mirror",
            "go": "removed_empty_generated_mirror",
            "reason": (
                "The modular bridge is Python orchestration. The maintained Rust authority is "
                "engine/src/quantum.rs and has its own parity/performance harness; generated "
                "Rust safety, Go, Julia, and Mojo bridge mirrors were nonfunctional."
            ),
        },
        "measurement_context": measurement_context(load_before),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and write canonical JSON evidence."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        evidence = run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"quantum-annealing benchmark written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
