# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — reproducible CLI cold-start benchmark

"""Compare cold CLI import and dispatch cost across two source trees."""

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

_SCHEMA_VERSION = "sc-neurocore.cli-startup-benchmark.v1"
_CHILD_PROGRAM = """
import contextlib
import io
import json
import resource
import sys
import time

started = time.perf_counter_ns()
from sc_neurocore.cli import main
import_ns = time.perf_counter_ns() - started
sys.argv = ["sc-neurocore", "--version"]
with contextlib.redirect_stdout(io.StringIO()):
    returncode = main()
rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
if sys.platform == "darwin":
    rss //= 1024
print(json.dumps({"import_ns": import_ns, "max_rss_kib": rss, "returncode": returncode}))
""".strip()


@dataclass(frozen=True)
class Variant:
    """One source checkout measured by the cold-start probe."""

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
    parser.add_argument(
        "--no-affinity",
        action="store_true",
        help="Do not prefix child probes with taskset",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid iteration counts, labels, roots, and CPU selections."""
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
        if not (root / "src" / "sc_neurocore").is_dir():
            raise ValueError(f"{label} root does not contain src/sc_neurocore")


def _source_digest(root: Path) -> str:
    """Hash the CLI source surface in either flat-module or package form."""
    flat_module = root / "src" / "sc_neurocore" / "cli.py"
    package = root / "src" / "sc_neurocore" / "cli"
    files = [flat_module] if flat_module.is_file() else sorted(package.rglob("*.py"))
    if not files:
        raise ValueError(f"no CLI source surface found below {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_metadata(variant: Variant) -> dict[str, object]:
    """Return path-free Git and source-integrity metadata for one variant."""
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
            "src/sc_neurocore/cli.py",
            "src/sc_neurocore/cli",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    git_head = head.stdout.strip() if head.returncode == 0 else None
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    reference = (
        f"git:{git_head}" if git_head is not None and not dirty else f"working-tree:{git_head}"
    )
    return {
        "label": variant.label,
        "source_root": reference,
        "local_path_recorded": False,
        "git_head": git_head,
        "cli_source_sha256": _source_digest(variant.root),
        "cli_surface_dirty": dirty,
    }


def _select_affinity(args: argparse.Namespace) -> tuple[int | None, str | None, str]:
    """Choose one allowed CPU and an optional taskset executable."""
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


def _sample(
    variant: Variant,
    *,
    cpu: int | None,
    taskset: str | None,
) -> dict[str, int]:
    """Run one fresh interpreter and validate its machine-readable probe output."""
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
    if not isinstance(payload, dict):
        raise RuntimeError(f"{variant.label} probe emitted a non-object payload")
    import_ns = payload.get("import_ns")
    max_rss_kib = payload.get("max_rss_kib")
    returncode = payload.get("returncode")
    if not isinstance(import_ns, int) or isinstance(import_ns, bool):
        raise RuntimeError(f"{variant.label} probe emitted non-integer metrics")
    if not isinstance(max_rss_kib, int) or isinstance(max_rss_kib, bool):
        raise RuntimeError(f"{variant.label} probe emitted non-integer metrics")
    if not isinstance(returncode, int) or isinstance(returncode, bool):
        raise RuntimeError(f"{variant.label} probe emitted non-integer metrics")
    if returncode != 0:
        raise RuntimeError(f"{variant.label} CLI returned {returncode}")
    return {
        "subprocess_wall_ns": wall_ns,
        "import_ns": import_ns,
        "max_rss_kib": max_rss_kib,
    }


def _statistics(samples: list[dict[str, int]]) -> dict[str, object]:
    """Summarise all recorded metrics without dropping raw samples."""
    summary: dict[str, object] = {"samples": samples, "sample_count": len(samples)}
    for metric in ("subprocess_wall_ns", "import_ns", "max_rss_kib"):
        values = [sample[metric] for sample in samples]
        summary[metric] = {
            "min": min(values),
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "max": max(values),
        }
    return summary


def _median(summary: dict[str, object], metric: str) -> float:
    """Read one validated median from an internal statistics payload."""
    metric_summary = summary[metric]
    if not isinstance(metric_summary, dict):
        raise RuntimeError(f"missing metric summary: {metric}")
    median = metric_summary.get("median")
    if not isinstance(median, (int, float)):
        raise RuntimeError(f"missing metric median: {metric}")
    return float(median)


def _comparison(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    """Compute candidate deltas relative to the parent source tree."""
    comparison: dict[str, object] = {}
    for metric in ("subprocess_wall_ns", "import_ns", "max_rss_kib"):
        baseline_median = _median(baseline, metric)
        candidate_median = _median(candidate, metric)
        comparison[metric] = {
            "baseline_median": baseline_median,
            "candidate_median": candidate_median,
            "candidate_minus_baseline": candidate_median - baseline_median,
            "candidate_delta_percent": (
                ((candidate_median / baseline_median) - 1.0) * 100.0
                if baseline_median != 0.0
                else None
            ),
        }
    return comparison


def run(args: argparse.Namespace) -> dict[str, object]:
    """Execute the interleaved parent/candidate benchmark and return its evidence."""
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
    evidence: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "local_cold_start_regression",
        "hardware_measurement_claimed": False,
        "operation": "import sc_neurocore.cli and dispatch main() with sys.argv=['--version']",
        "configuration": {
            "iterations": args.iterations,
            "warmups": args.warmups,
            "interleaving": "variant order reverses on alternating rounds",
            "python_executable": sys.executable,
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
                "python benchmarks/bench_cli_startup.py --baseline-root <parent-root> "
                "--candidate-root <candidate-root> --output <result.json>"
            ),
            "child": "[taskset --cpu-list <cpu>] <python> -c <cold-start probe>",
        },
        "variants": [_git_metadata(variant) for variant in variants],
        "results": results,
        "comparison": _comparison(results[variants[0].label], results[variants[1].label]),
        "polyglot_applicability": {
            "python": "measured",
            "rust": "not_applicable",
            "julia": "not_applicable",
            "mojo": "not_applicable",
            "go": "not_applicable",
            "reason": (
                "CLI parsing and process dispatch are Python-only; compute kernels retain their "
                "separate polyglot parity benchmarks"
            ),
        },
        "measurement_context": measurement_context(load_before),
    }
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark, write canonical JSON, and return a process status."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        evidence = run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CLI startup benchmark written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
