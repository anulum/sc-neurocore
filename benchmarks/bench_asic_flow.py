#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ASIC-flow modularisation benchmark

"""Compare deterministic ASIC deck generation across two source trees."""

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

_SCHEMA_VERSION = "sc-neurocore.asic-flow-modularisation-benchmark.v1"
_PROBE = Path(__file__).with_name("_asic_flow_probe.py")
_METRICS = (
    "subprocess_wall_ns",
    "import_ns",
    "deck_generation_ns",
    "bundle_write_ns",
    "max_rss_kib",
)


@dataclass(frozen=True)
class Variant:
    """One source tree measured by the cold-process probe."""

    label: str
    root: Path
    source_ref: str


def _parser() -> argparse.ArgumentParser:
    """Build the benchmark command parser."""
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
    """Reject invalid sample counts, labels, references, and source roots."""
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
        if not (root / "src/sc_neurocore/asic_flow/asic_flow.py").is_file():
            raise ValueError(f"{label} root has no ASIC-flow implementation")


def _source_files(root: Path) -> list[Path]:
    """Return every Python source file in the ASIC-flow package."""
    return sorted((root / "src/sc_neurocore/asic_flow").glob("*.py"))


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_digest(root: Path) -> tuple[str, int, dict[str, str]]:
    """Hash relative names and bytes for the imported ASIC-flow surface."""
    files = _source_files(root)
    if not files:
        raise ValueError(f"no ASIC-flow source found below {root}")
    digest = hashlib.sha256()
    hashes: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
        hashes[relative] = _sha256(path)
    return digest.hexdigest(), len(files), hashes


def _source_metadata(variant: Variant) -> dict[str, object]:
    """Return path-free source identity and integrity metadata."""
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
            "src/sc_neurocore/asic_flow",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    digest, file_count, hashes = _source_digest(variant.root)
    return {
        "label": variant.label,
        "source_ref": variant.source_ref,
        "local_path_recorded": False,
        "git_head": head.stdout.strip() if head.returncode == 0 else None,
        "surface_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "source_sha256": digest,
        "source_file_count": file_count,
        "source_hashes": hashes,
    }


def _select_affinity(args: argparse.Namespace) -> tuple[int | None, str | None, str]:
    """Choose an allowed CPU and describe whether affinity is exclusive."""
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


def _sample(variant: Variant, *, cpu: int | None, taskset: str | None) -> dict[str, int | str]:
    """Run and validate one cold-process ASIC deck probe."""
    command = [sys.executable, str(_PROBE)]
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
    if not isinstance(payload, dict) or payload.get("bundle_file_count") != 9:
        raise RuntimeError(f"{variant.label} probe emitted an invalid bundle payload")
    metrics: dict[str, int | str] = {"subprocess_wall_ns": wall_ns}
    for metric in _METRICS[1:]:
        value = payload.get(metric)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeError(f"{variant.label} probe emitted invalid {metric}")
        metrics[metric] = value
    for field in ("generated_sha256", "manifest_schema"):
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"{variant.label} probe emitted invalid {field}")
        metrics[field] = value
    generated_bytes = payload.get("generated_bytes")
    if not isinstance(generated_bytes, int) or generated_bytes <= 0:
        raise RuntimeError(f"{variant.label} probe emitted invalid generated_bytes")
    metrics["generated_bytes"] = generated_bytes
    return metrics


def _statistics(samples: list[dict[str, int | str]]) -> dict[str, object]:
    """Summarise timing metrics while retaining every raw sample."""
    summary: dict[str, object] = {"samples": samples, "sample_count": len(samples)}
    for metric in _METRICS:
        values = [sample[metric] for sample in samples]
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            raise RuntimeError(f"non-integer timing sample for {metric}")
        numeric = [int(value) for value in values]
        summary[metric] = {
            "min": min(numeric),
            "median": statistics.median(numeric),
            "mean": statistics.fmean(numeric),
            "max": max(numeric),
        }
    digests = {str(sample["generated_sha256"]) for sample in samples}
    byte_counts = {int(sample["generated_bytes"]) for sample in samples}
    schemas = {str(sample["manifest_schema"]) for sample in samples}
    if len(digests) != 1 or len(byte_counts) != 1 or len(schemas) != 1:
        raise RuntimeError("ASIC probe output changed between samples")
    summary["generated_sha256"] = digests.pop()
    summary["generated_bytes"] = byte_counts.pop()
    summary["manifest_schema"] = schemas.pop()
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
    """Compute candidate timing deltas and assert byte-identical outputs."""
    if baseline["generated_sha256"] != candidate["generated_sha256"]:
        raise RuntimeError("candidate ASIC decks differ from the baseline")
    comparison: dict[str, object] = {
        "generated_output_byte_identical": True,
        "generated_sha256": baseline["generated_sha256"],
        "generated_bytes": baseline["generated_bytes"],
    }
    for metric in _METRICS:
        baseline_median = _median(baseline, metric)
        candidate_median = _median(candidate, metric)
        comparison[metric] = {
            "baseline_median": baseline_median,
            "candidate_median": candidate_median,
            "candidate_minus_baseline": candidate_median - baseline_median,
            "candidate_delta_percent": (candidate_median / baseline_median - 1.0) * 100.0,
        }
    return comparison


def run(args: argparse.Namespace) -> dict[str, object]:
    """Execute an interleaved parent/candidate ASIC-flow benchmark."""
    _validate_args(args)
    variants = (
        Variant(
            str(args.baseline_label), Path(args.baseline_root).resolve(), str(args.baseline_ref)
        ),
        Variant(
            str(args.candidate_label), Path(args.candidate_root).resolve(), str(args.candidate_ref)
        ),
    )
    cpu, taskset, affinity_mode = _select_affinity(args)
    load_before = load_average()
    for warmup in range(args.warmups):
        ordered = variants if warmup % 2 == 0 else tuple(reversed(variants))
        for variant in ordered:
            _sample(variant, cpu=cpu, taskset=taskset)

    samples: dict[str, list[dict[str, int | str]]] = {variant.label: [] for variant in variants}
    for iteration in range(args.iterations):
        ordered = variants if iteration % 2 == 0 else tuple(reversed(variants))
        for variant in ordered:
            samples[variant.label].append(_sample(variant, cpu=cpu, taskset=taskset))
    results = {variant.label: _statistics(samples[variant.label]) for variant in variants}
    variant_metadata = [_source_metadata(variant) for variant in variants]
    candidate_metadata = variant_metadata[1]
    return {
        "schema_version": _SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "local_source_refactor_regression",
        "hardware_measurement_claimed": False,
        "external_eda_executed": False,
        "operation": "cold import, deterministic deck generation, and manifest-bearing bundle write",
        "configuration": {
            "iterations": args.iterations,
            "warmups": args.warmups,
            "deck_operations_per_sample": 25,
            "bundle_operations_per_sample": 3,
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
                "python benchmarks/bench_asic_flow.py --baseline-root <parent-root> "
                "--baseline-ref <parent-sha> --candidate-root <candidate-root> "
                "--output benchmarks/results/bench_asic_flow.json"
            ),
            "child": "[taskset --cpu-list <cpu>] <python> benchmarks/_asic_flow_probe.py",
        },
        "benchmark_sha256": _sha256(Path(__file__)),
        "probe_sha256": _sha256(_PROBE),
        "source_sha256": candidate_metadata["source_sha256"],
        "source_file_count": candidate_metadata["source_file_count"],
        "source_hashes": candidate_metadata["source_hashes"],
        "variants": variant_metadata,
        "results": results,
        "comparison": _comparison(results[variants[0].label], results[variants[1].label]),
        "polyglot_applicability": {
            "python": "measured_parent_candidate",
            "rust": "removed_nonfunctional_generated_mirror",
            "julia": "removed_non_parsing_generated_mirror",
            "go": "removed_empty_generated_mirror",
            "mojo": "removed_zero_return_generated_mirror",
            "reason": (
                "ASIC deck construction is typed text and filesystem orchestration. Its maintained "
                "execution boundary is external Yosys, OpenROAD, OpenSTA, KLayout, Magic, and Netgen, "
                "not an in-process numerical acceleration kernel."
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
    print(f"ASIC-flow benchmark written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
