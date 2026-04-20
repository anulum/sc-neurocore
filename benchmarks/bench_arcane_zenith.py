# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ArcaneZenith microbenchmark harness

"""Measures :class:`ArcaneZenithCognitiveCore.step` throughput on the
torch backend. Backs `docs/api/arcane_zenith.md` §7.1.
"""

import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))

from sc_neurocore.arcane_zenith import (  # noqa: E402
    create_arcane_neuron_with_zenith_plasticity,
)


def main() -> int:
    core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
    rng = np.random.default_rng(42)

    # Warmup (PyTorch autograd setup, dynamic graph construction, etc.)
    for _ in range(100):
        core.step(float(rng.uniform(-2, 5)))

    N = 5000
    t0 = time.perf_counter()
    for _ in range(N):
        core.step(float(rng.uniform(-2, 5)))
    dt = time.perf_counter() - t0

    results = {
        "arcane_zenith_step_torch": {
            "steps_per_s": N / dt,
            "us_per_step": 1e6 * dt / N,
            "identity_drift_after_5k_steps": core.neuron._identity_drift,
        }
    }

    print(f"\n{'Metric':<36} {'Value':>16}")
    print("-" * 54)
    print(f"{'ArcaneZenith.step torch':<36} {N/dt:>10.0f} steps/s")
    print(f"{'per-step latency':<36} {1e6*dt/N:>12.1f} µs")
    print(f"{'identity_drift (5k steps, τ_deep≥1s)':<36} {core.neuron._identity_drift:>16.4f}")

    out_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bench_arcane_zenith.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
