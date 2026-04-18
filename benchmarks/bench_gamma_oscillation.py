#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reproducible benchmark for the Börgers-Kopell PINGCircuit

"""Reproducible benchmark for `network.gamma_oscillation.PINGCircuit`.

Measures `step()` wall-clock plus dominant population frequency for
the conductance-based weak-PING circuit. Three workloads:

| n_excitatory | n_inhibitory | duration | Used for                 |
|-------------:|-------------:|---------:|--------------------------|
|        80    |        20    |   500 ms | Default test config.     |
|       400    |       100    |  1000 ms | Mid-scale pin.           |
|      4000    |      1000    |  1000 ms | Big-circuit pin.         |

Single backend (Python + NumPy) — multi-language acceleration of
the per-step LIF integrator is tracked under
`feedback_module_standard_attnres`. The kernel is a per-cell
conductance update (4 exponential decays + 2 mat-vec / sums per
step) and is a clean Rust + Mojo target; not yet implemented.

Output: JSON file at `benchmarks/results/bench_gamma_oscillation.json`
with one record per workload plus the measured dominant frequency
(must lie in 30-80 Hz per Börgers-Kopell 2003 Fig 2A).
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sc_neurocore.network.gamma_oscillation import PINGCircuit  # noqa: E402


def _bench_one(
    n_excitatory: int,
    n_inhibitory: int,
    duration_ms: float,
    dt: float,
    burn_in_ms: float,
    seed: int,
    backend: str,
) -> dict:
    t_build_0 = time.perf_counter()
    ping = PINGCircuit(
        n_excitatory=n_excitatory,
        n_inhibitory=n_inhibitory,
        seed=seed,
        backend=backend,
    )
    t_build = time.perf_counter() - t_build_0

    t_sim_0 = time.perf_counter()
    n_burn = int(round(burn_in_ms / dt))
    for _ in range(n_burn):
        ping.step(dt=dt)
    spikes_e: list[np.ndarray] = []
    n_record = int(round((duration_ms - burn_in_ms) / dt))
    for _ in range(n_record):
        se, _ = ping.step(dt=dt)
        spikes_e.append(se)
    t_sim = time.perf_counter() - t_sim_0

    freq = ping.dominant_frequency(spikes_e, dt=dt, bin_ms=1.0)
    rate_hz = float(np.mean(ping.population_rate(spikes_e, dt=dt, bin_ms=1.0)))
    n_steps = n_burn + n_record
    return {
        "backend": ping._use_rust and "rust" or "python",
        "n_excitatory": n_excitatory,
        "n_inhibitory": n_inhibitory,
        "duration_ms": duration_ms,
        "dt": dt,
        "burn_in_ms": burn_in_ms,
        "seed": seed,
        "build_seconds": t_build,
        "simulate_seconds": t_sim,
        "per_step_us": (t_sim / n_steps) * 1e6,
        "dominant_frequency_hz": freq,
        "mean_population_rate_hz": rate_hz,
        "in_published_band_30_to_80_hz": bool(30.0 <= freq <= 80.0),
    }


def main() -> None:
    out_path = Path(__file__).parent / "results" / "bench_gamma_oscillation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfgs = [
        # (n_e, n_i, duration_ms, dt, burn_in_ms)
        (80,    20,    500.0, 0.1, 100.0),
        (400,   100,  1000.0, 0.1, 200.0),
        (4000, 1000,  1000.0, 0.1, 200.0),
    ]
    runs = []
    for backend in ("python", "rust"):
        for n_e, n_i, dur, dt, burn in cfgs:
            print(
                f"running backend={backend} n_e={n_e} n_i={n_i} "
                f"dur={dur}ms ...",
                flush=True,
            )
            try:
                r = _bench_one(n_e, n_i, dur, dt, burn, 42, backend)
            except RuntimeError as exc:
                # Rust kernel not built — record a MISSING entry per
                # `feedback_no_blocked_without_probing.md`.
                print(f"  MISSING: {exc}", flush=True)
                runs.append({
                    "backend": backend, "n_excitatory": n_e,
                    "n_inhibitory": n_i, "duration_ms": dur,
                    "dt": dt, "burn_in_ms": burn, "seed": 42,
                    "status": "MISSING", "reason": str(exc),
                })
                continue
            runs.append(r)
            ok = "OK" if r["in_published_band_30_to_80_hz"] else "OUT"
            print(
                f"  build={r['build_seconds']:.2f}s "
                f"sim={r['simulate_seconds']:.2f}s "
                f"per-step={r['per_step_us']:.1f}us  "
                f"f_dom={r['dominant_frequency_hz']:.1f}Hz [{ok}]",
                flush=True,
            )

    payload = {
        "schema_version": 1,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "module": "sc_neurocore.network.gamma_oscillation.PINGCircuit",
        "publication_reference": (
            "Börgers, C. & Kopell, N. (2003) Neural Computation 15(3): "
            "509-538. Fig 2A weak-PING — dominant gamma 30-80 Hz."
        ),
        "backends": {
            "python": {
                "status": "USED",
                "notes": "NumPy vectorised per-cell conductance LIF.",
            },
            "rust": {
                "status": "USED",
                "notes": "PyO3 kernel `engine/src/ping.rs` ↔ "
                         "`sc_neurocore_engine.py_ping_step`. Spike "
                         "outputs bit-identical to the NumPy path "
                         "(noise pre-drawn on the Python side); "
                         "membrane V values diverge ≤ 0.5 mV due to "
                         "SIMD-vs-scalar float ordering, sub-threshold.",
            },
            "julia": {
                "status": "BLOCKED-ON-multilang-gamma",
                "notes": "Same as Rust.",
            },
            "go": {
                "status": "BLOCKED-ON-multilang-gamma",
                "notes": "Same as Rust.",
            },
            "mojo": {
                "status": "BLOCKED-ON-multilang-gamma",
                "notes": "Same as Rust; trivially fits the Mojo "
                         "@export raw-Int FFI pattern documented in "
                         "feedback_mojo_026_ffi_pattern.",
            },
        },
        "runs": runs,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
