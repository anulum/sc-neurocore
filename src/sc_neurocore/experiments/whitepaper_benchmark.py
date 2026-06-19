# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Whitepaper Benchmark

import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer
from sc_neurocore.profiling.energy import track_energy, profiler
from sc_neurocore.accel.jit_kernels import HAS_NUMBA


def run_whitepaper_benchmark() -> None:
    print("--- SC-NEUROCORE FOUNDATIONAL BENCHMARK ---")

    # Configuration
    N_INPUTS = 1000
    N_NEURONS = 1000
    LENGTH = 1024  # Bitstream length
    TRIALS = 10

    print(
        f"Configuration: Inputs={N_INPUTS}, Neurons={N_NEURONS}, Length={LENGTH}, JIT={HAS_NUMBA}"
    )

    # 1. Throughput & Latency (TPS)
    # We treat one forward pass of (1000x1000) as a "Block of Transactions"
    # Or 1 neuron update = 1 Op.
    # Total Ops = N_INPUTS * N_NEURONS * LENGTH

    layer = VectorizedSCLayer(n_inputs=N_INPUTS, n_neurons=N_NEURONS, length=LENGTH)

    # Warmup
    _ = layer.forward(np.random.random(N_INPUTS))  # type: ignore[arg-type]

    start_time = time.time()
    for _ in range(TRIALS):
        _ = layer.forward(np.random.random(N_INPUTS))  # type: ignore[arg-type]
    end_time = time.time()

    total_time = end_time - start_time
    avg_latency = total_time / TRIALS
    total_ops = N_INPUTS * N_NEURONS * LENGTH * TRIALS
    ops_per_sec = total_ops / total_time

    print("\n[Performance Results]")
    print(f"Average Latency (Forward Pass): {avg_latency * 1000:.2f} ms")
    print(f"Throughput (Bit-Ops/sec): {ops_per_sec:.2e}")
    # Equivalent "TPS" if 1 Tx = 256 ops?
    # Let's say 1 Tx = processing 1 input vector against the network state
    tps = (TRIALS) / total_time * N_INPUTS  # Inputs processed per second
    # No, TPS usually means "Ledger updates".
    # Let's define TPS as number of vector updates per second
    tps = 1.0 / avg_latency
    print(f"Layer Updates/Sec (Hz): {tps:.2f}")

    # 2. Energy Efficiency
    profiler.reset()
    # We decorate the method manually for the instance to capture this specific run
    # Note: 'track_energy' in our impl sums theoretical ops based on dimensions
    # It doesn't measure wall power (which is impossible in pure software)
    # But it gives us the 45nm equivalent.

    layer.forward = track_energy(layer.forward)  # type: ignore[method-assign]
    _ = layer.forward(np.random.random(N_INPUTS))

    joules = profiler.estimate_energy()
    co2 = profiler.co2_emission_g()

    # Ops in one pass
    ops_one_pass = N_INPUTS * N_NEURONS * LENGTH
    j_per_op = joules / ops_one_pass

    print("\n[Efficiency Results (45nm Simulation)]")
    print(f"Energy per Inference: {joules * 1e6:.2f} uJ")
    print(f"Energy per Bit-Op: {j_per_op * 1e15:.2f} fJ")
    print(f"CO2 Emissions per Inference: {co2:.2e} g")


if __name__ == "__main__":
    run_whitepaper_benchmark()
