# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — IR Compilation Demo

"""Build, verify, round-trip and emit synapse and Dense compute graphs.

Run with the installed Rust engine:
``python examples/03_ir_compile_demo.py --output-dir ./generated-ir``.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

try:
    from sc_neurocore_engine.ir import ScGraph, ScGraphBuilder, parse_ir
except ImportError:
    print(
        "This example requires the Rust engine. Install with: cd bridge && maturin develop --release"
    )
    sys.exit(0)


def build_synapse_graph() -> ScGraph:
    """Build a synapse with encoded probabilities, AND and Q8.8 rate decoding."""
    b = ScGraphBuilder("sc_synapse")

    # Inputs: two rate-coded probabilities
    x = b.input("x_prob", "rate")
    w = b.input("w_prob", "rate")

    # Encode to bitstreams
    x_bits = b.encode(x, length=1024, seed=0xACE1)
    w_bits = b.encode(w, length=1024, seed=0xBEEF)

    # SC multiply (AND)
    product = b.bitwise_and(x_bits, w_bits)

    # Extract firing rate
    rate = b.decode(product)

    # Output
    b.output("firing_rate", rate)

    return b.build()


def build_dense_graph() -> ScGraph:
    """Build a dense layer with 3 inputs, 7 neurons."""
    b = ScGraphBuilder("sc_dense_layer")

    x = b.input("x_input", "vec<rate,3>")
    w = b.input("weights", "vec<rate,21>")
    leak = b.input("leak_k", "i16")
    gain = b.input("gain_k", "i16")

    spikes = b.dense_forward(
        x,
        w,
        leak,
        gain,
        n_inputs=3,
        n_neurons=7,
        stream_length=1024,
    )

    b.output("spikes", spikes)
    return b.build()


def main() -> None:
    """Verify both graph examples and write emitted RTL to the selected directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=pathlib.Path(__file__).parent / "output"
    )
    args = parser.parse_args()
    print("SC-NeuroCore IR Compilation Demo")
    print("=" * 50)

    # -- Synapse graph --
    print("\n1. Building synapse graph...")
    synapse = build_synapse_graph()
    print(f"   Graph: {synapse}")
    print(f"   Inputs: {synapse.num_inputs}, Outputs: {synapse.num_outputs}")
    print(f"   Ops: {len(synapse)}")

    errors = synapse.verify()
    if errors is None:
        print("   Verification: PASS")
    else:
        print(f"   Verification FAILED: {errors}")
        return

    print("\n   Text format:")
    text = synapse.to_text()
    for line in text.strip().split("\n"):
        print(f"   | {line}")

    # Round-trip check
    parsed = parse_ir(text)
    assert parsed.to_text() == text, "Round-trip failed!"
    print("   Round-trip: PASS")

    # Emit SystemVerilog
    sv = synapse.emit_sv()
    print(f"\n   SystemVerilog: {len(sv)} characters")

    out_dir: pathlib.Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    synapse_path = out_dir / "generated_synapse.sv"
    synapse_path.write_text(sv, encoding="utf-8")
    print(f"   Wrote: {synapse_path}")

    # -- Dense layer graph --
    print("\n2. Building dense layer graph...")
    dense = build_dense_graph()
    print(f"   Graph: {dense}")
    print(f"   Inputs: {dense.num_inputs}, Outputs: {dense.num_outputs}")

    errors = dense.verify()
    if errors is None:
        print("   Verification: PASS")
    else:
        print(f"   Verification FAILED: {errors}")
        return

    sv_dense = dense.emit_sv()
    dense_path = out_dir / "generated_dense.sv"
    dense_path.write_text(sv_dense, encoding="utf-8")
    print(f"   SystemVerilog: {len(sv_dense)} characters")
    print(f"   Wrote: {dense_path}")

    print("\nDone. Generated HDL targets the modules in hdl/.")


if __name__ == "__main__":
    main()
