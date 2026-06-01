# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ArcaneNeuron exact-relaxation benchmark

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import statistics
import time

from sc_neurocore.neurons.models.arcane_neuron import ArcaneNeuron


STEPS = 50_000
REPEATS = 5
CURRENT = 2.0
OUTPUT = Path("benchmarks/results/local_python_2026-06-01_arcane_neuron.json")


def run_once() -> dict[str, object]:
    neuron = ArcaneNeuron()
    spikes = 0
    start_ns = time.perf_counter_ns()
    for _ in range(STEPS):
        spikes += neuron.step(CURRENT)
    elapsed_ns = time.perf_counter_ns() - start_ns
    return {
        "steps": STEPS,
        "current": CURRENT,
        "elapsed_ns": elapsed_ns,
        "ns_per_step": elapsed_ns / STEPS,
        "spikes": spikes,
        "ending_state": neuron.get_state(),
        "predictor_weights": [float(value) for value in neuron.w_pred],
    }


def main() -> int:
    results = [run_once() for _ in range(REPEATS)]
    ns_per_step = [float(result["ns_per_step"]) for result in results]
    report = {
        "benchmark": "ArcaneNeuron Python exact-relaxation candidate-first step",
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "command": "PYTHONPATH=src .venv/bin/python benchmarks/bench_model_arcane_neuron.py",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "steps": STEPS,
        "repeats": REPEATS,
        "current": CURRENT,
        "median_ns_per_step": statistics.median(ns_per_step),
        "min_ns_per_step": min(ns_per_step),
        "max_ns_per_step": max(ns_per_step),
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
