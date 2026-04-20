# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — evo_substrate microbenchmark harness

"""Measures the throughput of the evolutionary substrate hot paths.

Results are time.perf_counter deltas; no smoothing, no simulation.
Emits a markdown table to stdout and a JSON snapshot to
benchmarks/results/bench_evo_substrate.json for doc consumption.
"""

import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))

from sc_neurocore.evo_substrate.evo_substrate import (  # noqa: E402
    CrossoverEngine,
    FormalSafetyGuard,
    Genome,
    MutationEngine,
    Organism,
    ReplicationEngine,
    assign_species,
    genomic_distance,
)


def _ns_per_call(fn, iters: int) -> float:
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e9 / iters


def main() -> int:
    results: dict[str, dict[str, float]] = {}

    me = MutationEngine(rng_seed=7)
    seed_g = Genome()
    seed_g.compute_id()
    results["mutate"] = {"ns_per_call": _ns_per_call(lambda: me.mutate(seed_g), 10_000)}

    ce = CrossoverEngine(rng_seed=7)
    a = Genome()
    a.compute_id()
    b = Genome()
    b.compute_id()
    results["crossover"] = {
        "ns_per_call": _ns_per_call(lambda: ce.crossover(a, b), 10_000),
    }

    results["genomic_distance"] = {
        "ns_per_call": _ns_per_call(lambda: genomic_distance(a, b), 10_000),
    }

    sg = FormalSafetyGuard()
    results["safety_guard_check"] = {
        "ns_per_call": _ns_per_call(lambda: sg.check(seed_g), 10_000),
    }

    pop = [
        Organism(genome=Genome.from_vector(np.random.RandomState(i).rand(19) * 2, i))
        for i in range(64)
    ]
    for p in pop:
        p.genome.compute_id()
    results["assign_species_n64"] = {
        "ns_per_call": _ns_per_call(lambda: assign_species(pop, threshold=0.3), 1_000),
    }

    def metrics_fn(g):
        return {"accuracy": 0.5 + 0.01 * g.topology.num_neurons / 32}

    engine = ReplicationEngine(max_population=32, industrial_mode=True)
    for i in range(16):
        g = Genome()
        g.compute_id()
        engine.seed(g)
    engine.evaluate_all(metrics_fn)
    results["evolve_generation_pop32"] = {
        "ns_per_call": _ns_per_call(lambda: engine.evolve_generation(metrics_fn), 20),
    }

    print(f"\n{'Operation':<40} {'ns/call':>14} {'ops/s':>14}")
    print("-" * 72)
    for op, m in results.items():
        ns = m["ns_per_call"]
        ops = 1e9 / ns
        print(f"{op:<40} {ns:>14.1f} {ops:>14.0f}")

    out_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bench_evo_substrate.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
