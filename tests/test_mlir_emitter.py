# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for MLIR Emitter

from sc_neurocore.compiler.mlir_emitter import MLIREmitter


def test_mlir_emitter_basic():
    emitter = MLIREmitter("sc_neuron")

    # Simulate a small SC circuit: (X AND Y) MUX Z
    # Using LFSRs as sources
    lfsr1 = emitter.emit_lfsr(16, 0xACE1)
    lfsr2 = emitter.emit_lfsr(16, 0x1234)
    lfsr3 = emitter.emit_lfsr(16, 0x5678)

    and_node = emitter.emit_and(lfsr1, lfsr2)
    mux_node = emitter.emit_mux(lfsr3, and_node, lfsr1)

    mlir = emitter.generate()

    assert "hw.module @sc_neuron" in mlir
    assert "comb.and" in mlir
    assert "comb.mux" in mlir
    assert "hw.instance" in mlir
    assert "hw.output" in mlir

    # print(mlir)


if __name__ == "__main__":
    test_mlir_emitter_basic()
    print("MLIR Emitter test passed!")
