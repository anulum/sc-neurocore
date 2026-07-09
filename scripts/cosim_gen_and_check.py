# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

#!/usr/bin/env python3
"""
Co-simulation driver for sc_lif_neuron hardware verification.

Usage:
  1. Generate stimuli + expected results:
       python cosim_gen_and_check.py --generate

  2. Run Verilog simulation (external, reads stimuli.txt -> results_verilog.txt):
       iverilog -o tb_lif ../hdl/sc_lif_neuron.v ../hdl/tb_sc_lif_neuron.v
       vvp tb_lif

  3. Compare results:
       python cosim_gen_and_check.py --check
"""

import argparse
import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sc_neurocore.neurons.fixed_point_lif import FixedPointLIFNeuron


def generate_stimuli(num_steps: int = 1000, seed: int = 42) -> list:
    """Generate reproducible random stimuli for co-simulation."""
    rng = np.random.default_rng(seed)
    inputs = []
    leak_k = 20  # Small leak (20/256 ~ 0.078)
    gain_k = 256  # Gain = 1.0

    for i in range(num_steps):
        # Occasional current pulses (10% of cycles)
        if rng.random() < 0.1:
            I_t = int(rng.uniform(0.5, 2.0) * 256)
        else:
            I_t = 0

        noise_in = int(rng.normal(0, 0.1) * 256)
        inputs.append((leak_k, gain_k, I_t, noise_in))

    return inputs


def run_python_model(inputs):
    """Run the bit-true Python model and return (spike, v_out) per step."""
    neuron = FixedPointLIFNeuron()
    results = []
    for leak, gain, i_t, noise in inputs:
        spike, v = neuron.step(leak, gain, i_t, noise)
        results.append((spike, v))
    return results


def write_stimuli(inputs, path="stimuli.txt"):
    with open(path, "w") as f:
        for leak, gain, i_t, noise in inputs:
            f.write(f"{leak} {gain} {i_t} {noise}\n")
    print(f"Wrote {len(inputs)} stimuli to {path}")


def write_expected(results, path="results_expected.txt"):
    with open(path, "w") as f:
        for spike, v in results:
            f.write(f"{spike} {v}\n")
    print(f"Wrote {len(results)} expected results to {path}")


def check_results(expected, verilog_path="results_verilog.txt"):
    if not os.path.exists(verilog_path):
        print(f"ERROR: {verilog_path} not found. Run the Verilog simulation first.")
        return False

    with open(verilog_path, "r") as f:
        verilog_lines = f.readlines()

    if len(verilog_lines) != len(expected):
        print(f"Length mismatch: Expected {len(expected)} vs Verilog {len(verilog_lines)}")

    mismatches = 0
    for i, (line, (exp_spike, exp_v)) in enumerate(zip(verilog_lines, expected)):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        v_spike = int(parts[0])
        v_pot = int(parts[1])

        if v_spike != exp_spike or v_pot != exp_v:
            if mismatches < 10:
                print(
                    f"  Mismatch at step {i}: "
                    f"Expected (spike={exp_spike}, v={exp_v}) "
                    f"vs Verilog (spike={v_spike}, v={v_pot})"
                )
            mismatches += 1

    if mismatches == 0:
        print(f"SUCCESS: All {len(expected)} steps match bit-exactly!")
        return True
    else:
        print(f"FAILURE: {mismatches}/{len(expected)} mismatches found.")
        return False


def main():
    parser = argparse.ArgumentParser(description="SC-NeuroCore Co-Simulation Driver")
    parser.add_argument(
        "--generate", action="store_true", help="Generate stimuli + expected results"
    )
    parser.add_argument(
        "--check", action="store_true", help="Compare Verilog results against Python model"
    )
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = parser.parse_args()

    if not args.generate and not args.check:
        parser.print_help()
        print("\nQuick start:")
        print("  python cosim_gen_and_check.py --generate")
        print("  # Run Verilog sim (iverilog + vvp)")
        print("  python cosim_gen_and_check.py --check")
        return

    if args.generate:
        inputs = generate_stimuli(args.steps, args.seed)
        expected = run_python_model(inputs)
        write_stimuli(inputs)
        write_expected(expected)
        print(f"\nStimuli and expected results generated for {args.steps} steps.")
        print("Next: compile and run the Verilog testbench (tb_sc_lif_neuron.v)")

    if args.check:
        inputs = generate_stimuli(args.steps, args.seed)
        expected = run_python_model(inputs)
        success = check_results(expected)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
