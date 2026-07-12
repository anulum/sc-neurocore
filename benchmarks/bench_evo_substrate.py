# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary substrate benchmark evidence

"""Measure evolutionary-substrate workflows with source-bound raw samples."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

from _benchmark_context import load_average, measurement_context
from sc_neurocore.evo_substrate import (
    CrossoverEngine,
    FormalSafetyGuard,
    Genome,
    MutationEngine,
    Organism,
    ReplicationEngine,
    assign_species,
    genomic_distance,
)

_SCHEMA_VERSION = "sc-neurocore.evo-substrate-benchmark.v2"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT = _REPO_ROOT / "benchmarks" / "results" / "bench_evo_substrate.json"


@dataclass(frozen=True)
class Operation:
    """Describe one timed operation and its per-sample preparation."""

    name: str
    iterations: int
    prepare: Callable[[], Callable[[], object]]


def _parser() -> argparse.ArgumentParser:
    """Build the benchmark command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Reject non-positive sample counts and negative warm-up counts."""
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")


def _source_digest() -> tuple[str, int]:
    """Hash every Python file in the evolutionary-substrate package."""
    source_root = _REPO_ROOT / "src" / "sc_neurocore" / "evo_substrate"
    files = sorted(source_root.glob("*.py"))
    if not files:
        raise RuntimeError("evolutionary-substrate source package is empty")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(_REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), len(files)


def _git_metadata() -> dict[str, object]:
    """Return Git identity and scoped dirty-state metadata."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "src/sc_neurocore/evo_substrate",
            "benchmarks/bench_evo_substrate.py",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    digest, file_count = _source_digest()
    return {
        "git_head": head.stdout.strip() if head.returncode == 0 else None,
        "surface_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "source_sha256": digest,
        "source_file_count": file_count,
    }


def _assert_source_tree_import() -> None:
    """Reject measurements imported from an installed or stale source tree."""
    imported = Path(inspect.getfile(Genome)).resolve()
    expected = (_REPO_ROOT / "src" / "sc_neurocore" / "evo_substrate").resolve()
    if expected not in imported.parents:
        raise RuntimeError(f"Genome imported outside the source tree: {imported}")


def _mutation() -> Callable[[], object]:
    engine = MutationEngine(rng_seed=7)
    genome = Genome()
    genome.compute_id()
    return lambda: engine.mutate(genome)


def _crossover() -> Callable[[], object]:
    engine = CrossoverEngine(rng_seed=7)
    left = Genome()
    right = Genome()
    left.compute_id()
    right.compute_id()
    return lambda: engine.crossover(left, right)


def _distance() -> Callable[[], object]:
    left = Genome()
    right = Genome()
    right.topology.num_neurons = 32
    return lambda: genomic_distance(left, right)


def _safety() -> Callable[[], object]:
    guard = FormalSafetyGuard()
    genome = Genome()
    return lambda: guard.check(genome)


def _species() -> Callable[[], object]:
    population = [
        Organism(genome=Genome.from_vector(np.random.default_rng(index).random(19) * 2, index))
        for index in range(64)
    ]
    for organism in population:
        organism.genome.compute_id()
    return lambda: assign_species(population, threshold=0.3)


def _metrics(genome: Genome) -> dict[str, float]:
    """Return deterministic accuracy for the benchmark evolution workflow."""
    return {"accuracy": 0.5 + 0.01 * genome.topology.num_neurons / 32}


def _evolution() -> Callable[[], object]:
    engine = ReplicationEngine(max_population=32, industrial_mode=True)
    for _ in range(16):
        engine.seed(Genome())
    engine.evaluate_all(_metrics)
    return lambda: engine.evolve_generation(_metrics)


def _operations() -> tuple[Operation, ...]:
    """Return the fixed operation set and inner-iteration counts."""
    return (
        Operation("mutate", 1_000, _mutation),
        Operation("crossover", 1_000, _crossover),
        Operation("genomic_distance", 5_000, _distance),
        Operation("safety_guard_check", 5_000, _safety),
        Operation("assign_species_n64", 100, _species),
        Operation("evolve_generation_pop32", 10, _evolution),
    )


def _measure(operation: Operation) -> float:
    """Return nanoseconds per call for one freshly prepared sample."""
    function = operation.prepare()
    started = time.perf_counter_ns()
    for _ in range(operation.iterations):
        function()
    elapsed = time.perf_counter_ns() - started
    return elapsed / operation.iterations


def _summarise(samples: list[float], iterations: int) -> dict[str, object]:
    """Retain raw samples and compute descriptive timing statistics."""
    if not samples or any(not math.isfinite(value) or value <= 0.0 for value in samples):
        raise RuntimeError("benchmark produced an invalid timing sample")
    return {
        "iterations_per_sample": iterations,
        "sample_count": len(samples),
        "samples_ns_per_call": samples,
        "min_ns_per_call": min(samples),
        "median_ns_per_call": statistics.median(samples),
        "mean_ns_per_call": statistics.fmean(samples),
        "max_ns_per_call": max(samples),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    """Execute warm-ups and timed samples for every operation."""
    _validate_args(args)
    _assert_source_tree_import()
    load_before = load_average()
    operations = _operations()
    for _ in range(args.warmups):
        for operation in operations:
            _measure(operation)

    measured: dict[str, object] = {}
    for operation in operations:
        samples = [_measure(operation) for _ in range(args.samples)]
        measured[operation.name] = _summarise(samples, operation.iterations)

    return {
        "schema_version": _SCHEMA_VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "command": "PYTHONPATH=src taskset -c <cpu> .venv/bin/python benchmarks/bench_evo_substrate.py",
        "protocol": {
            "samples": args.samples,
            "warmups": args.warmups,
            "clock": "time.perf_counter_ns",
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "source": _git_metadata(),
        "measurement_context": measurement_context(load_before),
        "operations": measured,
    }


def _print_summary(payload: dict[str, object]) -> None:
    """Print median latency and throughput for each operation."""
    operations = payload["operations"]
    if not isinstance(operations, dict):
        raise RuntimeError("benchmark payload has no operations")
    print(f"\n{'Operation':<34} {'median ns/call':>18} {'median ops/s':>16}")
    print("-" * 72)
    for name, raw in operations.items():
        if not isinstance(raw, dict):
            raise RuntimeError(f"invalid operation summary: {name}")
        median = raw.get("median_ns_per_call")
        if not isinstance(median, (int, float)):
            raise RuntimeError(f"missing median for {name}")
        print(f"{name:<34} {median:>18.1f} {1e9 / median:>16.0f}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and write its JSON evidence file."""
    args = _parser().parse_args(argv)
    payload = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _print_summary(payload)
    print(f"Results written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
