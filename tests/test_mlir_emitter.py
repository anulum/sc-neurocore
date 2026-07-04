# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for MLIR Emitter

import json
from pathlib import Path

import pytest

from sc_neurocore.compiler.mlir_emitter import MLIREmitter, generate_mlir_bundle


def test_mlir_emitter_basic() -> None:
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


def test_mlir_emitter_emits_xor_operation() -> None:
    emitter = MLIREmitter("test_xor")
    lhs_and_rhs = emitter.emit_and("in1", "in2")

    emitter.emit_xor(lhs_and_rhs, "in1")

    assert "comb.xor" in emitter.generate()


def test_mlir_emitter_emits_mux_operation() -> None:
    emitter = MLIREmitter("test_mux")

    emitter.emit_mux("cond", "t", "f")

    assert "comb.mux" in emitter.generate()


def test_mlir_bundle_writes_manifest(tmp_path: Path) -> None:
    emitter = MLIREmitter("native_sc_top")
    lhs = emitter.emit_lfsr(8, 0x5A)
    rhs = emitter.emit_lfsr(8, 0xC3)
    emitter.emit_and(lhs, rhs)

    bundle = generate_mlir_bundle(emitter, tmp_path, firtool="definitely_missing_firtool")

    assert bundle.module_name == "native_sc_top"
    assert bundle.node_count == 3
    assert bundle.op_counts == {"comb.and": 1, "hw.instance": 2}
    assert bundle.firtool_path is None
    assert bundle.to_dict()["module_name"] == "native_sc_top"
    assert (tmp_path / "native_sc_top.mlir").is_file()
    assert (tmp_path / "mlir_bundle_manifest.json").is_file()

    manifest = json.loads((tmp_path / "mlir_bundle_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "sc-neurocore.mlir_bundle_manifest.v1"
    assert manifest["circt"]["available"] is False
    assert manifest["circt"]["executed"] is False
    assert manifest["claim_status"]["circt_lowering_executed"] is False
    assert manifest["claim_status"]["verilog_generated_from_mlir"] is False


def test_mlir_bundle_rejects_unsafe_module_name(tmp_path: Path) -> None:
    emitter = MLIREmitter("native-sc/top")
    emitter.emit_and("%lhs", "%rhs")

    with pytest.raises(ValueError, match="Invalid module name"):
        generate_mlir_bundle(emitter, tmp_path)


def test_mlir_bundle_method_rejects_implicit_external_execution(tmp_path: Path) -> None:
    emitter = MLIREmitter("safe_top")
    emitter.emit_xor("%a", "%b")

    with pytest.raises(NotImplementedError, match="CIRCT execution"):
        emitter.write_bundle(tmp_path, run_circt=True)


if __name__ == "__main__":
    test_mlir_emitter_basic()
    print("MLIR Emitter test passed!")
