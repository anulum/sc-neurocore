#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quadratic IF exact-flow local regression benchmark

from __future__ import annotations

import json
from pathlib import Path
import platform
import statistics
import time
from datetime import UTC, datetime
from typing import Any, Protocol

from sc_neurocore.neurons.models.quadratic_if import QuadraticIFNeuron


STEPS = 200_000
REPEATS = 5
CURRENT = 0.5
OUTPUT = Path("benchmarks/results/local_python_2026-06-16_quadratic_if_exact_flow.json")


class _StepNeuron(Protocol):
    def step(self, current: float) -> int: ...


def _python_neuron() -> _StepNeuron:
    return QuadraticIFNeuron()


def _run_once(factory: Any, backend: str) -> dict[str, object]:
    neuron = factory()
    spikes = 0
    start_ns = time.perf_counter_ns()
    for _ in range(STEPS):
        spikes += int(neuron.step(CURRENT))
    elapsed_ns = time.perf_counter_ns() - start_ns
    return {
        "backend": backend,
        "steps": STEPS,
        "current": CURRENT,
        "elapsed_ns": elapsed_ns,
        "ns_per_step": elapsed_ns / STEPS,
        "spikes": spikes,
    }


def _run_backend(name: str, factory: Any) -> dict[str, object]:
    results = [_run_once(factory, name) for _ in range(REPEATS)]
    ns_per_step = [float(result["ns_per_step"]) for result in results]
    return {
        "backend": name,
        "median_ns_per_step": statistics.median(ns_per_step),
        "min_ns_per_step": min(ns_per_step),
        "max_ns_per_step": max(ns_per_step),
        "results": results,
    }


def main() -> int:
    report = {
        "benchmark": "QuadraticIFNeuron exact constant-current flow step",
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "command": "PYTHONPATH=src ./.venv/bin/python benchmarks/bench_model_quadratic_if.py",
        "evidence_class": "local_regression_non_isolated",
        "production_speed_claim": False,
        "isolation": "non-isolated loaded workstation",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "steps": STEPS,
        "repeats": REPEATS,
        "current": CURRENT,
        "results": [
            _run_backend("python", _python_neuron),
            {
                "backend": "rust",
                "skipped": True,
                "reason": "covered by focused Rust exact-flow tests; not loaded through this Python benchmark",
            },
            {
                "backend": "go",
                "skipped": True,
                "reason": "covered by focused Go exact-flow tests; not loaded through Python FFI",
            },
            {
                "backend": "julia",
                "skipped": True,
                "reason": "covered by Julia source mirror check; not loaded by this Python benchmark",
            },
            {
                "backend": "mojo",
                "skipped": True,
                "reason": "non-authoritative Mojo kernel mirror aligned; no shared library benchmark hook",
            },
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
