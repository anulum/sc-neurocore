# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

#!/usr/bin/env python3
"""Compare complete sc_lif_neuron RTL traces with the fixed-point Python model.

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
from collections.abc import Sequence
import os
import re
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sc_neurocore.neurons.fixed_point_lif import FixedPointLIFNeuron


def generate_stimuli(num_steps: int = 1000, seed: int = 42) -> list[tuple[int, int, int, int]]:
    """Generate seeded Q8 stimuli for the maintained LIF testbench.

    Parameters
    ----------
    num_steps : int, optional
        Positive number of applied clock steps.
    seed : int, optional
        Nonnegative NumPy random-generator seed.

    Returns
    -------
    list of tuple of int
        Leak, gain, input current and noise for each step, in Q8 units.

    Raises
    ------
    ValueError
        If the step count is nonpositive or the seed is negative.
    """
    if num_steps <= 0 or seed < 0:
        raise ValueError("steps must be positive and seed must be nonnegative")
    rng = np.random.default_rng(seed)
    inputs: list[tuple[int, int, int, int]] = []
    leak_k = 20
    gain_k = 256

    for _ in range(num_steps):
        if rng.random() < 0.1:
            I_t = int(rng.uniform(0.5, 2.0) * 256)
        else:
            I_t = 0

        noise_in = int(rng.normal(0, 0.1) * 256)
        inputs.append((leak_k, gain_k, I_t, noise_in))

    return inputs


def run_python_model(inputs: Sequence[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    """Execute the default signed 16-bit, Q8 fixed-point LIF reference.

    Parameters
    ----------
    inputs : sequence of tuple of int
        Leak, gain, current and noise values for successive clock steps.

    Returns
    -------
    list of tuple of int
        Binary spike and signed Q8 membrane potential for every input step.
    """
    neuron = FixedPointLIFNeuron()
    results = []
    for leak, gain, i_t, noise in inputs:
        spike, v = neuron.step(leak, gain, i_t, noise)
        results.append((spike, v))
    return results


def write_stimuli(inputs: Sequence[tuple[int, int, int, int]], path: str = "stimuli.txt") -> None:
    """Write the four decimal fields consumed by the Verilog testbench.

    Parameters
    ----------
    inputs : sequence of tuple of int
        Leak, gain, current and noise values for each clock step.
    path : str, optional
        Output stimulus file, resolved from the caller's working directory.

    Raises
    ------
    OSError
        If the stimulus file cannot be written.
    """
    with open(path, "w") as f:
        for leak, gain, i_t, noise in inputs:
            f.write(f"{leak} {gain} {i_t} {noise}\n")
    print(f"Wrote {len(inputs)} stimuli to {path}")


def write_expected(results: Sequence[tuple[int, int]], path: str = "results_expected.txt") -> None:
    """Write the reference spike and potential for each applied step.

    Parameters
    ----------
    results : sequence of tuple of int
        Binary spikes and signed Q8 membrane potentials.
    path : str, optional
        Output reference file, resolved from the caller's working directory.

    Raises
    ------
    OSError
        If the reference file cannot be written.
    """
    with open(path, "w") as f:
        for spike, v in results:
            f.write(f"{spike} {v}\n")
    print(f"Wrote {len(results)} expected results to {path}")


def check_results(
    expected: Sequence[tuple[int, int]], verilog_path: str = "results_verilog.txt"
) -> bool:
    """Require a complete, well-formed RTL trace matching the reference.

    Parameters
    ----------
    expected : sequence of tuple of int
        Nonempty reference trace of binary spikes and signed Q8 potentials.
    verilog_path : str, optional
        ASCII trace containing exactly two decimal fields per clock step.

    Returns
    -------
    bool
        True only when every expected step has one valid, bit-exact RTL row.
        Missing, unreadable, malformed or incomplete traces return False.
    """
    if not expected:
        print("ERROR: The reference trace must contain at least one step.")
        return False
    try:
        with open(verilog_path, "r", encoding="ascii") as f:
            verilog_lines = f.readlines()
    except (OSError, UnicodeError):
        print("ERROR: Cannot read the ASCII Verilog trace. Run the simulation first.")
        return False

    if len(verilog_lines) != len(expected):
        print(f"Length mismatch: Expected {len(expected)} vs Verilog {len(verilog_lines)}")
        return False

    mismatches = 0
    for i, (line, (exp_spike, exp_v)) in enumerate(zip(verilog_lines, expected)):
        parts = line.strip().split()
        if (
            len(parts) != 2
            or re.fullmatch(r"[01]", parts[0]) is None
            or re.fullmatch(r"[+-]?[0-9]+", parts[1]) is None
        ):
            print(f"ERROR: Malformed Verilog row at step {i}; expected spike and potential.")
            return False
        try:
            v_spike = int(parts[0])
            v_pot = int(parts[1])
        except ValueError:
            print(f"ERROR: Decimal value at step {i} cannot be read.")
            return False

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


def main(argv: Sequence[str] | None = None) -> int:
    """Generate reference inputs or validate the caller's RTL result file.

    Parameters
    ----------
    argv : sequence of str, optional
        CLI arguments; process arguments are used when omitted.

    Returns
    -------
    int
        Zero for generation, help or an exact match; one for a refused trace.

    Raises
    ------
    SystemExit
        With status two when CLI arguments are invalid.
    """
    parser = argparse.ArgumentParser(description="SC-NeuroCore Co-Simulation Driver")
    parser.add_argument(
        "--generate", action="store_true", help="Generate stimuli + expected results"
    )
    parser.add_argument(
        "--check", action="store_true", help="Compare Verilog results against Python model"
    )
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")

    if not args.generate and not args.check:
        parser.print_help()
        print("\nQuick start:")
        print("  python cosim_gen_and_check.py --generate")
        print("  # Run Verilog sim (iverilog + vvp)")
        print("  python cosim_gen_and_check.py --check")
        return 0

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
        return 0 if success else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
