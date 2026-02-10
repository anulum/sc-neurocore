"""
SC-NeuroCore IR Compilation Demo
=================================

Builds an SC compute graph from Python, verifies it,
prints its text representation, and emits synthesizable
SystemVerilog targeting the existing HDL modules.

Usage:
    cd 03_CODE/sc-neurocore
    $env:PYTHONPATH='src'
    .\\.venv\\Scripts\\python examples/02_ir_compile_demo.py
"""

from __future__ import annotations

import pathlib

from sc_neurocore_engine.ir import ScGraphBuilder, parse_ir


def build_synapse_graph() -> "ScGraph":
    """Build a minimal synapse: encode two probabilities, AND, popcount."""
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
    count = b.popcount(product)
    rate = b.div_const(count, 1024)

    # Output
    b.output("firing_rate", rate)

    return b.build()


def build_dense_graph() -> "ScGraph":
    """Build a dense layer with 3 inputs, 7 neurons."""
    b = ScGraphBuilder("sc_dense_layer")

    x = b.input("x_input", "rate")
    w = b.input("weights", "rate")
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

    out_dir = pathlib.Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
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
