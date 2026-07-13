#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-bound autonomous-learning benchmark

"""Compare deterministic autonomous-learning paths across two source trees."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

from _benchmark_context import load_average, measurement_context
from _learning_benchmark_support import (
    PROBE,
    Variant,
    compare,
    metadata,
    sample,
    select_affinity,
    summarize,
    validate_args,
)

_SCHEMA = "sc-neurocore.learning-bridge-modularisation-benchmark.v1"
_SUPPORT = Path(__file__).with_name("_learning_benchmark_support.py")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline-lib", type=Path, required=True)
    parser.add_argument("--candidate-lib", type=Path, required=True)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--candidate-ref", default="working-tree")
    parser.add_argument("--baseline-label", default="parent")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--no-affinity", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Return source-bound, interleaved parent/candidate evidence."""
    validate_args(args)
    variants = (
        Variant(
            args.baseline_label,
            args.baseline_root.resolve(),
            args.baseline_lib.resolve(),
            args.baseline_ref,
        ),
        Variant(
            args.candidate_label,
            args.candidate_root.resolve(),
            args.candidate_lib.resolve(),
            args.candidate_ref,
        ),
    )
    cpu, taskset, affinity = select_affinity(args)
    load_before = load_average()
    for index in range(args.warmups):
        for variant in variants if index % 2 == 0 else reversed(variants):
            sample(variant, args.steps, cpu, taskset)
    samples: dict[str, list[dict[str, Any]]] = {variant.label: [] for variant in variants}
    for index in range(args.iterations):
        for variant in variants if index % 2 == 0 else reversed(variants):
            samples[variant.label].append(sample(variant, args.steps, cpu, taskset))
    results = {variant.label: summarize(samples[variant.label]) for variant in variants}
    variants_metadata = [metadata(variant) for variant in variants]
    candidate = variants_metadata[1]
    return {
        "schema_version": _SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "local_source_and_native_library_regression",
        "hardware_measurement_claimed": False,
        "operation": "deterministic STDP scalar, batched, layer, Torch, Go, and Julia paths",
        "configuration": {
            "iterations": args.iterations,
            "warmups": args.warmups,
            "steps": args.steps,
            "interleaving": "variant order reverses on alternating rounds",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": cpu,
            "affinity_mode": affinity,
        },
        "commands": {
            "harness": (
                "python benchmarks/bench_autonomous_learning.py --baseline-root <root> "
                "--baseline-lib <lib> --baseline-ref <sha> --candidate-lib <lib> "
                "--output <json>"
            ),
            "child": (
                "[taskset --cpu-list <cpu>] <python> benchmarks/_learning_bridge_probe.py "
                "--root <root> --steps <n>"
            ),
        },
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "support_sha256": hashlib.sha256(_SUPPORT.read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest(),
        "source_sha256": candidate["source_sha256"],
        "source_file_count": candidate["source_file_count"],
        "source_hashes": candidate["source_hashes"],
        "variants": variants_metadata,
        "results": results,
        "comparison": compare(results[variants[0].label], results[variants[1].label]),
        "polyglot_applicability": {
            "python": "measured",
            "rust": "measured",
            "torch": "measured",
            "go": "measured_when_executable_available",
            "julia": "measured_when_executable_available",
        },
        "measurement_context": measurement_context(load_before),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and write its evidence JSON."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        evidence = run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Autonomous-learning benchmark written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
