#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-bound Bioware modularisation benchmark

"""Compare deterministic Bioware pipelines across two Python source trees."""

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
from typing import Any

from _benchmark_context import load_average, measurement_context

_SCHEMA = "sc-neurocore.bioware-modularisation-benchmark.v1"
_PROBE = Path(__file__).with_name("_bioware_probe.py")
_METRICS = ("subprocess_wall_ns", "import_ns", "pipeline_ns", "max_rss_kib")
_COUNTS = ("spike_count", "aer_event_count", "bitstream_count", "opto_pulse_count")


@dataclass(frozen=True)
class Variant:
    """One source tree measured by the isolated probe."""

    label: str
    root: Path
    source_ref: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--candidate-ref", default="working-tree")
    parser.add_argument("--baseline-label", default="parent")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--no-affinity", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if args.baseline_label == args.candidate_label:
        raise ValueError("baseline and candidate labels must differ")
    if not args.baseline_ref or not args.candidate_ref:
        raise ValueError("source references must be non-empty")
    for label, root in (
        (args.baseline_label, args.baseline_root),
        (args.candidate_label, args.candidate_root),
    ):
        if not (root / "src/sc_neurocore/bioware/bioware.py").is_file():
            raise ValueError(f"{label} root has no Bioware implementation")


def _source_digest(root: Path) -> tuple[str, dict[str, str]]:
    files = sorted((root / "src/sc_neurocore/bioware").glob("*.py"))
    if not files:
        raise ValueError(f"no Bioware source found below {root}")
    digest = hashlib.sha256()
    hashes: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        hashes[relative] = hashlib.sha256(content).hexdigest()
    return digest.hexdigest(), hashes


def _source_metadata(variant: Variant) -> dict[str, Any]:
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
            "src/sc_neurocore/bioware",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    digest, hashes = _source_digest(variant.root)
    return {
        "label": variant.label,
        "source_ref": variant.source_ref,
        "local_path_recorded": False,
        "git_head": head.stdout.strip() if head.returncode == 0 else None,
        "surface_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "source_sha256": digest,
        "source_file_count": len(hashes),
        "source_hashes": hashes,
    }


def _select_affinity(args: argparse.Namespace) -> tuple[int | None, str | None, str]:
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


def _sample(variant: Variant, *, cpu: int | None, taskset: str | None) -> dict[str, Any]:
    command = [sys.executable, str(_PROBE)]
    if cpu is not None and taskset is not None:
        command = [taskset, "--cpu-list", str(cpu), *command]
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(variant.root / "src")
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command,
        cwd=variant.root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
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
    if not isinstance(payload, dict):
        raise RuntimeError(f"{variant.label} probe emitted an invalid payload")
    sample: dict[str, Any] = {"subprocess_wall_ns": wall_ns}
    for field in (*_METRICS[1:], "canonical_bytes", *_COUNTS):
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeError(f"{variant.label} probe emitted invalid {field}")
        sample[field] = value
    digest = payload.get("canonical_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"{variant.label} probe emitted invalid canonical_sha256")
    sample["canonical_sha256"] = digest
    return sample


def _statistics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"samples": samples, "sample_count": len(samples)}
    for metric in _METRICS:
        values = [int(sample[metric]) for sample in samples]
        summary[metric] = {
            "min": min(values),
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "max": max(values),
        }
    for field in ("canonical_sha256", "canonical_bytes", *_COUNTS):
        stable_values = {sample[field] for sample in samples}
        if len(stable_values) != 1:
            raise RuntimeError(f"Bioware probe changed {field} between samples")
        summary[field] = stable_values.pop()
    return summary


def _comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    for field in ("canonical_sha256", "canonical_bytes", *_COUNTS):
        if baseline[field] != candidate[field]:
            raise RuntimeError(f"candidate Bioware output differs from baseline: {field}")
    comparison: dict[str, Any] = {
        "canonical_output_byte_identical": True,
        "canonical_sha256": baseline["canonical_sha256"],
        "canonical_bytes": baseline["canonical_bytes"],
        **{field: baseline[field] for field in _COUNTS},
    }
    for metric in _METRICS:
        baseline_median = float(baseline[metric]["median"])
        candidate_median = float(candidate[metric]["median"])
        comparison[metric] = {
            "baseline_median": baseline_median,
            "candidate_median": candidate_median,
            "candidate_minus_baseline": candidate_median - baseline_median,
            "candidate_delta_percent": (candidate_median / baseline_median - 1.0) * 100.0,
        }
    return comparison


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Measure alternating parent and candidate Bioware workflows.

    Parameters
    ----------
    args : argparse.Namespace
        Validated parser-produced benchmark configuration.

    Returns
    -------
    dict[str, Any]
        Source-bound measurements and deterministic output comparison.

    Raises
    ------
    OSError
        If a source tree, executable, or system measurement cannot be read.
    RuntimeError
        If the probe fails or parent and candidate outputs differ.
    ValueError
        If the requested benchmark configuration is invalid.
    """
    _validate_args(args)
    variants = (
        Variant(str(args.baseline_label), args.baseline_root.resolve(), str(args.baseline_ref)),
        Variant(str(args.candidate_label), args.candidate_root.resolve(), str(args.candidate_ref)),
    )
    cpu, taskset, affinity_mode = _select_affinity(args)
    load_before = load_average()
    for warmup in range(args.warmups):
        order = variants if warmup % 2 == 0 else tuple(reversed(variants))
        for variant in order:
            _sample(variant, cpu=cpu, taskset=taskset)
    samples: dict[str, list[dict[str, Any]]] = {variant.label: [] for variant in variants}
    for iteration in range(args.iterations):
        order = variants if iteration % 2 == 0 else tuple(reversed(variants))
        for variant in order:
            samples[variant.label].append(_sample(variant, cpu=cpu, taskset=taskset))
    results = {variant.label: _statistics(samples[variant.label]) for variant in variants}
    metadata = [_source_metadata(variant) for variant in variants]
    candidate = metadata[1]
    return {
        "schema_version": _SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "local_source_refactor_regression",
        "hardware_measurement_claimed": False,
        "external_eda_executed": False,
        "operation": "cold import and deterministic MEA-to-opto pipeline",
        "configuration": {
            "iterations": args.iterations,
            "warmups": args.warmups,
            "pipeline_operations_per_sample": 12,
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
                "python benchmarks/bench_bioware.py --baseline-root <parent-root> "
                "--baseline-ref <parent-sha> --candidate-root <candidate-root> "
                "--output benchmarks/results/bench_bioware.json"
            ),
            "child": "[taskset --cpu-list <cpu>] <python> benchmarks/_bioware_probe.py",
        },
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(_PROBE.read_bytes()).hexdigest(),
        "source_sha256": candidate["source_sha256"],
        "source_file_count": candidate["source_file_count"],
        "source_hashes": candidate["source_hashes"],
        "variants": metadata,
        "results": results,
        "comparison": _comparison(results[variants[0].label], results[variants[1].label]),
        "polyglot_applicability": {
            "python": "measured_parent_candidate",
            "rust": "not_applicable",
            "julia": "not_applicable",
            "go": "not_applicable",
            "mojo": "not_applicable",
            "reason": (
                "The maintained Bioware orchestration is Python-only. Removed generated "
                "return-zero and invalid-syntax mirrors had no executable dispatch path; "
                "independent Julia plasticity solvers remain outside this full-pipeline benchmark."
            ),
        },
        "measurement_context": measurement_context(load_before),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Bioware parent/candidate benchmark command.

    Parameters
    ----------
    argv : Sequence[str] or None, optional
        Arguments without the executable; ``None`` reads the process command line.

    Returns
    -------
    int
        Zero after writing the benchmark evidence JSON.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        evidence = run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Bioware benchmark written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
