# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — optics subsystem microbenchmark harness

"""Measures analyze_bank, analyze_pairs, and FDTD2D hot paths.

Backs `docs/api/optics.md` §7. Requires the Rust engine to be
importable (``maturin develop --release`` in a venv that
``pip install -e``-s the package); otherwise only the Python-fallback
figures will be reported.
"""

import json
import os
import random
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))


def main() -> int:
    from sc_neurocore.optics.photonic_emitter import (  # noqa: E402
        CrosstalkModel,
        FDTD2DSolver,
    )
    import sc_neurocore.optics.photonic_emitter as mod  # noqa: E402

    results: dict[str, dict[str, float]] = {}

    # 7.1 analyze_bank — 1000 back-to-back calls, Rust vs Python
    cm = CrosstalkModel()
    for _ in range(10):
        cm.analyze_bank(waveguides=100, gap_nm=200.0, coupling_length_um=10.0)
    N = 1000
    t0 = time.perf_counter()
    for _ in range(N):
        cm.analyze_bank(waveguides=100, gap_nm=200.0, coupling_length_um=10.0)
    dt_rust = time.perf_counter() - t0
    rust_bank_per_s = N / dt_rust

    orig = mod._HAS_RUST_PH
    mod._HAS_RUST_PH = False
    t0 = time.perf_counter()
    for _ in range(N):
        CrosstalkModel().analyze_bank(waveguides=100, gap_nm=200.0, coupling_length_um=10.0)
    dt_py = time.perf_counter() - t0
    py_bank_per_s = N / dt_py
    mod._HAS_RUST_PH = orig

    results["analyze_bank_rust"] = {
        "calls_per_s": rust_bank_per_s,
        "ns_per_call": dt_rust * 1e9 / N,
    }
    results["analyze_bank_python"] = {
        "calls_per_s": py_bank_per_s,
        "ns_per_call": dt_py * 1e9 / N,
    }
    results["analyze_bank_speedup"] = {"rust_over_python": rust_bank_per_s / py_bank_per_s}

    # 7.2 analyze_pairs — 5000 pairs, best-of-5 after warmup
    random.seed(42)
    pairs = [(i, i + 1) for i in range(5000)]
    gaps = [random.uniform(100, 600) for _ in range(5000)]
    lens = [random.uniform(5, 50) for _ in range(5000)]

    for _ in range(3):
        cm.analyze_pairs(pairs, gaps, lens)
    ts_rust = []
    for _ in range(5):
        t0 = time.perf_counter()
        cm.analyze_pairs(pairs, gaps, lens)
        ts_rust.append(time.perf_counter() - t0)
    rust_pairs_ms = min(ts_rust) * 1000

    orig = mod._HAS_RUST_PH
    mod._HAS_RUST_PH = False
    ts_py = []
    for _ in range(5):
        t0 = time.perf_counter()
        CrosstalkModel().analyze_pairs(pairs, gaps, lens)
        ts_py.append(time.perf_counter() - t0)
    py_pairs_ms = min(ts_py) * 1000
    mod._HAS_RUST_PH = orig

    results["analyze_pairs_rust"] = {"wall_ms_best_of_5": rust_pairs_ms}
    results["analyze_pairs_python"] = {"wall_ms_best_of_5": py_pairs_ms}
    results["analyze_pairs_speedup"] = {"rust_over_python": py_pairs_ms / rust_pairs_ms}

    # 7.3 FDTD2D — 500 steps on 200×100 grid after 50-step warmup
    s = FDTD2DSolver(nx=200, ny=100, pml_layers=12)
    s.set_waveguide(y_center=50, width_cells=10, refractive_index=3.48)
    s.inject_source(x=50, y=50, wavelength_nm=1550.0, amplitude=1.0, sigma_cells=8)
    s.step(50)
    t0 = time.perf_counter()
    s.step(500)
    dt = time.perf_counter() - t0
    cells = 200 * 100 * 500
    results["fdtd2d_200x100_500steps"] = {
        "wall_ms": dt * 1000,
        "mcell_steps_per_s": cells / dt / 1e6,
        "field_energy": s.field_energy(),
    }

    print(f"\n{'Benchmark':<34} {'Value':>22}")
    print("-" * 58)
    print(f"{'analyze_bank rust':<34} {rust_bank_per_s:>16.0f} calls/s")
    print(f"{'analyze_bank python':<34} {py_bank_per_s:>16.0f} calls/s")
    print(f"{'analyze_bank speedup':<34} {rust_bank_per_s / py_bank_per_s:>21.2f}x")
    print(f"{'analyze_pairs rust':<34} {rust_pairs_ms:>19.3f} ms")
    print(f"{'analyze_pairs python':<34} {py_pairs_ms:>19.3f} ms")
    print(f"{'analyze_pairs speedup':<34} {py_pairs_ms / rust_pairs_ms:>21.2f}x")
    print(f"{'fdtd2d 500 steps':<34} {dt * 1000:>16.1f} ms, {cells / dt / 1e6:.1f} Mcell-steps/s")

    out_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bench_optics.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
