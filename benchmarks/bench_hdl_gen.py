# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — hdl_gen microbenchmark harness

"""Measures Python-side RTL / SPICE emission throughput.

Emits stdout markdown + benchmarks/results/bench_hdl_gen.json.
Does NOT run yosys — that is driven by run_asic_flow.sh.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))

from sc_neurocore.hdl_gen import SpiceGenerator, VerilogGenerator  # noqa: E402


def _ns_per_call(fn, iters: int) -> float:
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e9 / iters


def main() -> int:
    results: dict[str, dict[str, float]] = {}

    gen = VerilogGenerator(module_name="my_sc_net_top")
    gen.add_layer("Dense", "l1", {"n_neurons": 32})
    gen.add_layer("Dense", "l2", {"n_neurons": 32})
    gen.add_layer("Dense", "l3", {"n_neurons": 10})
    results["verilog_generate_3layer"] = {
        "ns_per_call": _ns_per_call(gen.generate, 10_000),
    }

    W16 = np.random.default_rng(7).random((16, 16))

    def _spice_16():
        with tempfile.NamedTemporaryFile(suffix=".sp", delete=False) as tmp:
            path = tmp.name
        SpiceGenerator.generate_crossbar(W16, path)
        os.unlink(path)

    results["spice_generate_crossbar_16x16"] = {
        "ns_per_call": _ns_per_call(_spice_16, 500),
    }

    W64 = np.random.default_rng(7).random((64, 64))

    def _spice_64():
        with tempfile.NamedTemporaryFile(suffix=".sp", delete=False) as tmp:
            path = tmp.name
        SpiceGenerator.generate_crossbar(W64, path)
        os.unlink(path)

    results["spice_generate_crossbar_64x64"] = {
        "ns_per_call": _ns_per_call(_spice_64, 100),
    }

    if subprocess.run(["which", "yosys"], capture_output=True).returncode == 0:
        sv_path = os.path.join(
            SCRIPT_DIR,
            "..",
            "src",
            "sc_neurocore",
            "hdl_gen",
            "safety",
            "safety_monitor.sv",
        )
        t0 = time.perf_counter()
        proc = subprocess.run(
            [
                "yosys",
                "-q",
                "-p",
                f"read_verilog -sv {sv_path}; hierarchy -top neuro_safe_monitor; synth; stat",
            ],
            capture_output=True,
        )
        t1 = time.perf_counter()
        results["yosys_synth_safety_monitor"] = {
            "ns_per_call": (t1 - t0) * 1e9,
            "exit_code": float(proc.returncode),
        }

    print(f"\n{'Operation':<38} {'ns/call':>14} {'ops/s':>14}")
    print("-" * 70)
    for op, m in results.items():
        ns = m["ns_per_call"]
        ops = 1e9 / ns
        print(f"{op:<38} {ns:>14.1f} {ops:>14.2f}")

    out_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bench_hdl_gen.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
