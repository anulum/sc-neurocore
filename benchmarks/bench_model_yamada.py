#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — YamadaNeuron RK4 benchmark

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import statistics
import time

from sc_neurocore.neurons.models.yamada import YamadaNeuron


def _run_once(steps: int, current: float) -> dict[str, float | int | list[float]]:
    neuron = YamadaNeuron()
    spikes = 0
    start_ns = time.perf_counter_ns()
    for _ in range(steps):
        spikes += neuron.step(current)
    elapsed_ns = time.perf_counter_ns() - start_ns
    return {
        "steps": steps,
        "current": current,
        "elapsed_ns": elapsed_ns,
        "ns_per_step": elapsed_ns / steps,
        "spikes": spikes,
        "ending_state": [neuron.v, neuron.n, neuron.q],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark YamadaNeuron candidate-first RK4 step.")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--current", type=float, default=50.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    results = [_run_once(args.steps, args.current) for _ in range(args.repeats)]
    ns_per_step = [float(result["ns_per_step"]) for result in results]
    payload = {
        "benchmark": "YamadaNeuron Python RK4 step",
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "PYTHONPATH=src .venv/bin/python benchmarks/bench_model_yamada.py",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "steps": args.steps,
        "repeats": args.repeats,
        "current": args.current,
        "median_ns_per_step": statistics.median(ns_per_step),
        "min_ns_per_step": min(ns_per_step),
        "max_ns_per_step": max(ns_per_step),
        "results": results,
    }

    output = args.output
    if output is None:
        output = Path("benchmarks/results/local_python_2026-06-01_yamada.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
