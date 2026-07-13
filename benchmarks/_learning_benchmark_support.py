# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Autonomous-learning benchmark support

"""Validation, sampling, and comparison helpers for the learning benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any

PROBE = Path(__file__).with_name("_learning_bridge_probe.py")
TIMINGS = (
    "subprocess_wall_ns",
    "import_ns",
    "rust_scalar_ns",
    "rust_batched_ns",
    "rust_layer_ns",
    "torch_ns",
    "go_process_ns",
    "julia_process_ns",
)
METRICS = (*TIMINGS, "max_rss_kib")
HASHES = ("layer_weights_sha256", "layer_state_sha256")
WEIGHTS = (
    "rust_scalar_weight",
    "rust_batched_weight",
    "torch_weight",
    "go_weight",
    "julia_weight",
)


@dataclass(frozen=True)
class Variant:
    """One source tree and native library measured by the probe."""

    label: str
    root: Path
    library: Path
    source_ref: str


def validate_args(args: argparse.Namespace) -> None:
    """Reject incomplete or ambiguous benchmark configurations."""
    if args.iterations <= 0 or args.steps <= 0:
        raise ValueError("iterations and steps must be positive")
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if args.baseline_label == args.candidate_label:
        raise ValueError("baseline and candidate labels must differ")
    if not args.baseline_ref or not args.candidate_ref:
        raise ValueError("source references must be non-empty")
    pairs = ((args.baseline_root, args.baseline_lib), (args.candidate_root, args.candidate_lib))
    for root, library in pairs:
        if not (root / "src/sc_neurocore/_native/learning_bridge.py").is_file():
            raise ValueError(f"source root has no learning bridge: {root}")
        if not library.is_file():
            raise ValueError(f"native learning library does not exist: {library}")


def source_digest(root: Path) -> tuple[str, dict[str, str]]:
    """Hash every maintained autonomous-learning implementation surface."""
    files = list((root / "src/sc_neurocore/_native").glob("learning*.py"))
    relatives = (
        "crates/autonomous_learning/src/lib.rs",
        "crates/autonomous_learning/src/state_codec.rs",
        "crates/autonomous_learning/src/wgpu_backend.rs",
        "src/sc_neurocore/accel/go/autonomous_learning/learning_bridge.go",
        "src/sc_neurocore/accel/julia/_native/learning_bridge.jl",
    )
    files.extend(path for relative in relatives if (path := root / relative).is_file())
    if not files:
        raise ValueError(f"no autonomous-learning source found below {root}")
    digest = hashlib.sha256()
    hashes: dict[str, str] = {}
    for path in sorted(files):
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        digest.update(relative.encode() + b"\0" + content + b"\0")
        hashes[relative] = hashlib.sha256(content).hexdigest()
    return digest.hexdigest(), hashes


def metadata(variant: Variant) -> dict[str, Any]:
    """Describe one measured source tree and native library without local paths."""
    head = subprocess.run(
        ["git", "-C", str(variant.root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    digest, hashes = source_digest(variant.root)
    status = subprocess.run(
        ["git", "-C", str(variant.root), "status", "--porcelain", "--", *hashes],
        capture_output=True,
        text=True,
        check=False,
    )
    library = variant.library.read_bytes()
    return {
        "label": variant.label,
        "source_ref": variant.source_ref,
        "git_head": head.stdout.strip() if head.returncode == 0 else None,
        "surface_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "source_sha256": digest,
        "source_file_count": len(hashes),
        "source_hashes": hashes,
        "library_sha256": hashlib.sha256(library).hexdigest(),
        "library_bytes": len(library),
        "local_paths_recorded": False,
    }


def select_affinity(args: argparse.Namespace) -> tuple[int | None, str | None, str]:
    """Select a valid taskset CPU when affinity is enabled and supported."""
    if args.no_affinity:
        return None, None, "disabled by --no-affinity"
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    cpu = args.cpu if args.cpu is not None else (allowed[0] if allowed else None)
    if cpu is not None and allowed and cpu not in allowed:
        raise ValueError(f"cpu {cpu} is outside the process affinity set")
    taskset = shutil.which("taskset")
    if cpu is None or taskset is None:
        return None, None, "taskset or an allowed CPU was unavailable"
    return cpu, taskset, "taskset affinity pinned; CPU not exclusively isolated"


def sample(variant: Variant, steps: int, cpu: int | None, taskset: str | None) -> dict[str, Any]:
    """Run and validate one isolated multi-language probe sample."""
    command = [sys.executable, str(PROBE), "--root", str(variant.root), "--steps", str(steps)]
    if cpu is not None and taskset is not None:
        command = [taskset, "--cpu-list", str(cpu), *command]
    environment = os.environ.copy()
    environment.update(
        PYTHONHASHSEED="0",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(variant.root / "src"),
        SC_NEUROCORE_LIB_PATH=str(variant.library),
        LD_LIBRARY_PATH=str(variant.library.parent),
    )
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command, cwd=variant.root, env=environment, capture_output=True, text=True, timeout=180
    )
    wall_ns = time.perf_counter_ns() - started
    if completed.returncode != 0:
        raise RuntimeError(f"{variant.label} probe failed: {completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{variant.label} probe emitted invalid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("timings"), dict):
        raise RuntimeError(f"{variant.label} probe emitted an invalid payload")
    timings = payload["timings"]
    result: dict[str, Any] = {"subprocess_wall_ns": wall_ns}
    for metric in TIMINGS[1:]:
        value = timings.get(metric)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise RuntimeError(f"{variant.label} probe emitted invalid {metric}")
        result[metric] = value
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError(f"{variant.label} probe emitted invalid outputs")
    for field in WEIGHTS:
        value = outputs.get(field)
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise RuntimeError(f"{variant.label} probe emitted invalid {field}")
        result[field] = value
    for field in HASHES:
        value = outputs.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"{variant.label} probe emitted invalid {field}")
        result[field] = value
    digest = payload.get("canonical_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"{variant.label} probe emitted invalid canonical_sha256")
    result["canonical_sha256"] = digest
    for field in ("canonical_bytes", "max_rss_kib"):
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeError(f"{variant.label} probe emitted invalid {field}")
        result[field] = value
    return result


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate timings while requiring deterministic outputs."""
    summary: dict[str, Any] = {"samples": samples, "sample_count": len(samples)}
    for metric in METRICS:
        values = [int(item[metric]) for item in samples if item[metric] is not None]
        summary[metric] = {"available_samples": len(values)}
        if values:
            summary[metric].update(
                min=min(values),
                median=statistics.median(values),
                mean=statistics.fmean(values),
                max=max(values),
            )
    for field in ("canonical_sha256", "canonical_bytes", *HASHES, *WEIGHTS):
        values = {item[field] for item in samples}
        if len(values) != 1:
            raise RuntimeError(f"learning probe changed {field} between samples")
        summary[field] = values.pop()
    return summary


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Require equivalent outputs and calculate available timing deltas."""
    for field in ("canonical_sha256", "canonical_bytes", *HASHES):
        if baseline[field] != candidate[field]:
            raise RuntimeError(f"candidate autonomous-learning output differs: {field}")
    comparison: dict[str, Any] = {"canonical_output_equivalent": True}
    for metric in METRICS:
        left, right = baseline[metric], candidate[metric]
        if not left.get("available_samples") or not right.get("available_samples"):
            comparison[metric] = {"available": False}
            continue
        base, cand = float(left["median"]), float(right["median"])
        comparison[metric] = {
            "available": True,
            "baseline_median": base,
            "candidate_median": cand,
            "candidate_delta_percent": (cand / base - 1.0) * 100.0,
        }
    return comparison
