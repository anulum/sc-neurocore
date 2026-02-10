#!/usr/bin/env python3
"""Example 03: High-Performance Vectorized Layer.

Demonstrates the VectorizedSCLayer which uses packed 64-bit bitwise
AND + popcount for high throughput computation.
"""

import time
import numpy as np
from sc_neurocore import VectorizedSCLayer

def main():
    print("=== SC-NeuroCore: Vectorized Layer Performance ===\n")

    n_inputs = 32
    n_neurons = 16
    length = 2048

    layer = VectorizedSCLayer(
        n_inputs=n_inputs,
        n_neurons=n_neurons,
        length=length,
        use_gpu=False,  # CPU path for portability
    )

    input_probs = np.random.uniform(0.0, 1.0, n_inputs).tolist()

    # Warm-up
    layer.forward(input_probs)

    # Benchmark
    n_runs = 100
    start = time.perf_counter()
    for _ in range(n_runs):
        output = layer.forward(input_probs)
    elapsed = time.perf_counter() - start

    print(f"Config: {n_inputs} inputs x {n_neurons} neurons x {length}-bit streams")
    print(f"Output (last run): {output}")
    print(f"Throughput: {n_runs / elapsed:.1f} forward passes/sec")
    print(f"Latency: {elapsed / n_runs * 1000:.2f} ms/pass")
    print("\nDone.")


if __name__ == "__main__":
    main()
