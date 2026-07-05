# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CIRCT round-trip validation of the MLIR emitter

"""Round-trip the MLIR emitter output through the real CIRCT ``circt-opt``.

These tests prove the emitter produces genuinely verifiable ``hw``/``comb``
dialect MLIR — not merely MLIR-shaped text — by feeding it to ``circt-opt
--verify-diagnostics`` and lowering it to Verilog with ``circt-opt
--export-verilog``. They self-skip when ``circt-opt`` is not installed, mirroring
the ``g++`` gate in ``tests/test_hls_export.py``, so the suite stays green on
hosts without CIRCT while becoming a real hardware-lowering check where CIRCT is
present.

Install CIRCT (bundles ``circt-opt``) from the ``llvm/circt`` ``firtool-*``
releases and put its ``bin`` on ``PATH`` to exercise these locally.
"""

import shutil
import subprocess

import pytest

from sc_neurocore.compiler.mlir_emitter import MLIREmitter

_CIRCT_OPT = shutil.which("circt-opt")
requires_circt = pytest.mark.skipif(_CIRCT_OPT is None, reason="circt-opt not installed")

# 0xACE1 == 44257 — the Verilog exporter renders the parameter as decimal.
_SEED_ACE1 = 0xACE1


def _sc_pipeline() -> MLIREmitter:
    """Build a representative stochastic-computing pipeline.

    Three seeded LFSR bitstreams feed a ``comb.and`` (SC multiply), a
    ``comb.xor``, and a ``comb.mux`` (SC scaled add) — exercising every
    operation the emitter supports plus multi-instance uniqueness.
    """
    emitter = MLIREmitter("sc_neuron")
    lfsr_a = emitter.emit_lfsr(16, _SEED_ACE1)
    lfsr_b = emitter.emit_lfsr(16, 0x1234)
    lfsr_c = emitter.emit_lfsr(16, 0x5678)
    prod = emitter.emit_and(lfsr_a, lfsr_b)
    emitter.emit_xor(prod, lfsr_a)
    emitter.emit_mux(lfsr_c, prod, lfsr_a)
    return emitter


def _circt(args: list[str], mlir_text: str, tmp_path) -> subprocess.CompletedProcess[str]:
    source = tmp_path / "module.mlir"
    source.write_text(mlir_text, encoding="utf-8")
    return subprocess.run(  # nosec B603 - circt-opt path from shutil.which, literal flags
        [_CIRCT_OPT, *args, str(source)],
        capture_output=True,
        text=True,
        check=False,
    )


@requires_circt
def test_pipeline_mlir_verifies(tmp_path):
    """The emitted MLIR passes ``circt-opt`` parse + verifier with no diagnostics."""
    result = _circt(["--verify-diagnostics"], _sc_pipeline().generate(), tmp_path)
    assert result.returncode == 0, result.stderr


@requires_circt
def test_pipeline_lowers_to_verilog(tmp_path):
    """``circt-opt --export-verilog`` lowers the hw/comb module to real Verilog."""
    result = _circt(["--export-verilog"], _sc_pipeline().generate(), tmp_path)
    assert result.returncode == 0, result.stderr
    verilog = result.stdout
    assert "module sc_neuron(" in verilog
    assert "endmodule" in verilog
    # The parametric extern lowers to a parameterised instantiation.
    assert "sc_lfsr" in verilog
    assert ".WIDTH(16)" in verilog
    assert f".SEED({_SEED_ACE1})" in verilog
    # The SC dataflow survives lowering: AND for multiply, ternary for the mux.
    assert "&" in verilog and "?" in verilog


@requires_circt
def test_empty_pipeline_verifies(tmp_path):
    """An empty pipeline still emits a verifiable module driving its output."""
    result = _circt(["--verify-diagnostics"], MLIREmitter("empty").generate(), tmp_path)
    assert result.returncode == 0, result.stderr


@requires_circt
def test_instance_names_are_unique(tmp_path):
    """Each LFSR gets a distinct instance symbol (CIRCT rejects duplicates)."""
    mlir = _sc_pipeline().generate()
    for name in ('"lfsr1"', '"lfsr2"', '"lfsr3"'):
        assert mlir.count(name) == 1, f"{name} must appear exactly once"
    # Prove the uniqueness matters: the module verifies precisely because the
    # names differ — a regression to a shared name would fail circt-opt.
    assert _circt(["--verify-diagnostics"], mlir, tmp_path).returncode == 0


@requires_circt
def test_missing_extern_declaration_is_rejected(tmp_path):
    """Adversarial: an instance of an undeclared module must fail the verifier.

    This guards the emitter's extern-declaration contract — if a future change
    dropped the ``hw.module.extern`` line, ``circt-opt`` would reject the output,
    and this test proves the gate has teeth.
    """
    mlir = _sc_pipeline().generate()
    without_extern = "\n".join(
        line for line in mlir.splitlines() if not line.startswith("hw.module.extern")
    )
    result = _circt(["--verify-diagnostics"], without_extern, tmp_path)
    assert result.returncode != 0


def test_generate_declares_extern_before_top():
    """Regression (no CIRCT needed): every instantiated module is declared."""
    mlir = _sc_pipeline().generate()
    extern_index = mlir.index("hw.module.extern @sc_lfsr")
    top_index = mlir.index("hw.module @sc_neuron")
    assert extern_index < top_index
    assert "WIDTH: i32, SEED: i32" in mlir
    assert "-> (out: i1)" in mlir
