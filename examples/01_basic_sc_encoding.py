#!/usr/bin/env python3
"""Example 01: Basic Stochastic Computing Encoding/Decoding.

Demonstrates how to encode scalar values into bitstreams and decode them back,
using both Bernoulli and Sobol (low-discrepancy) methods.
"""

from sc_neurocore import BitstreamEncoder, bitstream_to_probability

def main():
    print("=== SC-NeuroCore: Basic Bitstream Encoding ===\n")

    # Bernoulli encoding (random)
    encoder_b = BitstreamEncoder(x_min=0.0, x_max=1.0, length=1024, mode="bernoulli")
    for target in [0.1, 0.3, 0.5, 0.7, 0.9]:
        bs = encoder_b.encode(target)
        recovered = bitstream_to_probability(bs)
        error = abs(recovered - target)
        print(f"  Bernoulli: target={target:.1f}  recovered={recovered:.4f}  error={error:.4f}")

    print()

    # Sobol encoding (low-discrepancy — faster convergence)
    encoder_s = BitstreamEncoder(x_min=0.0, x_max=1.0, length=1024, mode="sobol", seed=42)
    for target in [0.1, 0.3, 0.5, 0.7, 0.9]:
        bs = encoder_s.encode(target)
        recovered = bitstream_to_probability(bs)
        error = abs(recovered - target)
        print(f"  Sobol:     target={target:.1f}  recovered={recovered:.4f}  error={error:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
