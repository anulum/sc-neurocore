#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reproducible benchmark for the Potjans CorticalColumn

"""Reproducible benchmark for `network.cortical_column.CorticalColumn`.

Measures wall-clock for `simulate(duration_ms, dt)` across the three
useful Python operating points:

| Configuration                                  | Used for             |
|------------------------------------------------|----------------------|
| `scale=0.02, scale_correction=False`           | Smoke / determinism. |
| `scale=0.05, scale_correction=True`            | Mid-scale fidelity.  |
| `scale=0.1,  scale_correction=True`            | Published-fidelity.  |

Single backend (Python + scipy.sparse) — the multi-language
acceleration chain (Rust + Julia + Go + Mojo) for the per-step
inner loop is tracked as a separate follow-up under the
`feedback_module_standard_attnres` policy. The chain has not been
implemented yet because the inner loop is a 64-way sparse mat-vec
product whose proper acceleration requires a block-sparse
restructure of the connectivity. Documenting that gap honestly is
preferable to shipping a fake "accelerated" path.

Output: JSON file at `benchmarks/results/bench_cortical_column.json`
with one record per (configuration, repetition) plus the published
firing rates for cross-verification.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

# Make `python benchmarks/bench_cortical_column.py` work without
# having to set PYTHONPATH manually — matches the pattern used by
# `dna_mapper_benchmark.py` and `stochastic_doctor_benchmark.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sc_neurocore.network.cortical_column import (  # noqa: E402  — sys.path is mutated above on purpose
    POPULATIONS,
    CorticalColumn,
)

# Per Potjans 2014 Table 4 — the published asynchronous-irregular
# rates the benchmark cross-checks against (informational, not a
# parity contract because rates are emergent and the present
# implementation matches L4e exactly but sits 2-4× above the
# others).
POTJANS_TABLE4_HZ: dict[str, float] = {
    "L23e": 0.86,
    "L23i": 2.91,
    "L4e": 4.51,
    "L4i": 5.78,
    "L5e": 7.59,
    "L5i": 8.13,
    "L6e": 1.10,
    "L6i": 8.07,
}


def _bench_one(
    scale: float,
    scale_correction: bool,
    duration_ms: float,
    dt: float,
    burn_in_ms: float,
    seed: int,
    delay_distribution: bool = True,
) -> dict:
    """Build, simulate, measure wall-clock and per-pop rates."""
    t_build_0 = time.perf_counter()
    col = CorticalColumn(
        scale=scale,
        scale_correction=scale_correction,
        seed=seed,
        delay_distribution=delay_distribution,
    )
    t_build = time.perf_counter() - t_build_0

    t_sim_0 = time.perf_counter()
    rasters = col.simulate(duration_ms=duration_ms, dt=dt)
    t_sim = time.perf_counter() - t_sim_0

    rates = col.population_rates(
        rasters,
        dt=dt,
        burn_in_ms=burn_in_ms,
    )
    n_steps = int(round(duration_ms / dt))
    return {
        "scale": scale,
        "scale_correction": scale_correction,
        "delay_distribution": delay_distribution,
        "duration_ms": duration_ms,
        "dt": dt,
        "burn_in_ms": burn_in_ms,
        "seed": seed,
        "n_total": col.n_total,
        "sizes": dict(col.sizes),
        "build_seconds": t_build,
        "simulate_seconds": t_sim,
        "per_step_ms": (t_sim / n_steps) * 1e3,
        "rates_hz": dict(rates),
        "potjans_table4_hz": dict(POTJANS_TABLE4_HZ),
        "rate_ratio_to_table4": {
            p: (rates[p] / POTJANS_TABLE4_HZ[p]) if POTJANS_TABLE4_HZ[p] > 0 else None
            for p in POPULATIONS
        },
    }


def main() -> None:
    out_path = Path(__file__).parent / "results" / "bench_cortical_column.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfgs = [
        # (scale, scale_correction, duration_ms, dt, burn_in_ms,
        #  delay_distribution)
        (0.02, False, 100.0, 0.1, 20.0, False),
        (0.05, True, 300.0, 0.1, 100.0, False),
        (0.10, True, 600.0, 0.1, 200.0, False),
        # Per-connection Gaussian delays — slower (5x more sparse
        # mat-vecs per step) but rate-fidelity dramatically tighter.
        (0.10, True, 600.0, 0.1, 200.0, True),
    ]
    runs = []
    for scale, sc, dur, dt, burn, dd in cfgs:
        print(
            f"running scale={scale} corr={sc} dist={dd} dur={dur}ms ...",
            flush=True,
        )
        r = _bench_one(scale, sc, dur, dt, burn, 42, dd)
        runs.append(r)
        print(
            f"  build={r['build_seconds']:.2f}s "
            f"sim={r['simulate_seconds']:.2f}s "
            f"per-step={r['per_step_ms']:.2f}ms",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "module": "sc_neurocore.network.cortical_column.CorticalColumn",
        "publication_reference": (
            "Potjans & Diesmann (2014) Cerebral Cortex 24(3):785-806, "
            "DOI 10.1093/cercor/bhs358 — Table 4 firing rates."
        ),
        "backends": {
            "python": {
                "status": "USED",
                "notes": "scipy.sparse CSR per (target,source) pair.",
            },
            "rust": {
                "status": "BLOCKED-ON-multilang-cortical",
                "notes": "Per-step inner loop is 64-way sparse mat-vec "
                "across population pairs; proper acceleration "
                "needs a block-sparse restructure of the "
                "connectivity. Tracked as follow-up under "
                "feedback_module_standard_attnres.",
            },
            "julia": {
                "status": "BLOCKED-ON-multilang-cortical",
                "notes": "Same as Rust.",
            },
            "go": {
                "status": "BLOCKED-ON-multilang-cortical",
                "notes": "Same as Rust.",
            },
            "mojo": {
                "status": "BLOCKED-ON-multilang-cortical",
                "notes": "Same as Rust.",
            },
        },
        "runs": runs,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
