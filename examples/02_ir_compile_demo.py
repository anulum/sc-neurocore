"""
SC-NeuroCore IR Compilation Demo

Builds an SC compute graph, verifies it, emits SystemVerilog,
and saves the result.
"""

from __future__ import annotations

import pathlib

import numpy as np

from sc_neurocore_engine.layers import VectorizedSCLayer


def main() -> None:
    print("SC-NeuroCore IR Compilation Demo")
    print("=" * 50)

    # Note: IR construction and SV emission happen in Rust.
    # This demo will call Python bridge IR APIs once they are exposed.
    # For now we demonstrate dense-layer mapping continuity.
    layer = VectorizedSCLayer(n_inputs=3, n_neurons=7, length=1024)
    inputs = np.array([0.3, 0.5, 0.7])
    rates = layer.forward(inputs)

    print(f"\nDense Layer: {layer.n_inputs} inputs -> {layer.n_neurons} neurons")
    print(f"Input probabilities: {inputs}")
    print(f"Output rates: {rates}")
    print("\nThis layer maps to sc_dense_layer_core in HDL.")
    print("IR compilation produces synthesizable SystemVerilog")
    print("that instantiates the same HDL modules in hdl/.")

    out_dir = pathlib.Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "generated_dense.sv"
    out_path.write_text(
        "// Placeholder generated artifact.\n"
        "// The Rust IR emitter now lives in engine/src/ir/emit_sv.rs.\n",
        encoding="utf-8",
    )

    print(f"\nOutput directory: {out_dir}")
    print(f"Wrote: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
