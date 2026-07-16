# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — LeakyCompeteFire exact-relaxation benchmark

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import time
from typing import cast

from sc_neurocore.neurons.models.leaky_compete_fire import LeakyCompeteFireNeuron


STEPS = 50_000
REPEATS = 5
CURRENTS = [5.0, 2.5, 1.25, 0.5]
OUTPUT = Path("benchmarks/results/local_python_2026-06-01_leaky_compete_fire.json")


def run_once() -> dict[str, object]:
    neuron = LeakyCompeteFireNeuron()
    total_spikes = 0
    start_ns = time.perf_counter_ns()
    for _ in range(STEPS):
        total_spikes += sum(neuron.step(CURRENTS))
    elapsed_ns = time.perf_counter_ns() - start_ns
    return {
        "steps": STEPS,
        "currents": CURRENTS,
        "elapsed_ns": elapsed_ns,
        "ns_per_step": elapsed_ns / STEPS,
        "total_spikes": total_spikes,
        "ending_state": list(neuron.v),
    }


def main() -> int:
    results = [run_once() for _ in range(REPEATS)]
    ns_per_step = [cast(float, result["ns_per_step"]) for result in results]
    report = {
        "spdx_license": "AGPL-3.0-or-later",
        "commercial_license": "available",
        "copyright_concepts": "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "copyright_code": "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "orcid": "0009-0009-3560-0851",
        "contact": "www.anulum.li | protoscience@anulum.li",
        "project": "SC-NeuroCore",
        "benchmark": "LeakyCompeteFireNeuron Python exact-relaxation WTA step",
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "command": "PYTHONPATH=src .venv/bin/python benchmarks/bench_model_leaky_compete_fire.py",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "steps": STEPS,
        "repeats": REPEATS,
        "currents": CURRENTS,
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
