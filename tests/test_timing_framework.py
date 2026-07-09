# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Timing-aware formal property framework workflow contract tests.

This file verifies the NEU-C.2 workflow contract: a bounded timing property is
represented once, emitted deterministically for external model-checker surfaces,
and connected to a concrete dense-layer formal proof without mocking the RTL
unit under test.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "hdl" / "formal"))

from timing import (  # noqa: E402
    TimingProofOrchestrator,
    TimingProperty,
    emit_kind2_module,
    emit_nuxmv_module,
)

TIMING_DIR = REPO_ROOT / "hdl" / "formal" / "timing"
EXAMPLE_SBY = TIMING_DIR / "example_dense_layer_core_latency.sby"
CDC_SBY = TIMING_DIR / "example_cdc_two_flop_synchroniser.sby"
CDC_SV = TIMING_DIR / "example_cdc_two_flop_synchroniser.sv"
WRAPPER_LIB = TIMING_DIR / "timing_wrapper_lib.sv"
ASSERTIONS_SVH = TIMING_DIR / "timing_assertions.svh"

_FORMAL_UNAVAILABLE = shutil.which("sby") is None or shutil.which("cvc5") is None


def test_emitters_encode_bounded_timing_contract() -> None:
    """Both external model-checker emitters encode the bounded timing contract."""
    prop = TimingProperty(
        name="dense_start_to_done",
        kind="latency",
        trigger="start_pulse",
        response="run_done",
        bound_cycles=6,
    )

    nuxmv_model = emit_nuxmv_module(prop)
    kind2_model = emit_kind2_module(prop)

    assert "MODULE main" in nuxmv_model
    assert "age : 0..6" in nuxmv_model
    assert "INVARSPEC !violation" in nuxmv_model
    assert "node dense_start_to_done" in kind2_model
    assert "--%PROPERTY ok;" in kind2_model
    assert "pre_age >= 6" in kind2_model


def test_timing_property_rejects_invalid_bounds() -> None:
    """A negative cycle bound is rejected at property construction."""
    with pytest.raises(ValueError, match="bound_cycles"):
        TimingProperty(
            name="dense_bad_bound",
            kind="deadline",
            trigger="start_pulse",
            response="run_done",
            bound_cycles=-1,
        )


def test_orchestrator_reports_missing_external_dependency() -> None:
    """A missing formal executable fails the proof instead of passing it."""
    orchestrator = TimingProofOrchestrator(
        EXAMPLE_SBY,
        executable="sc_neurocore_missing_sby",
        solver="sc_neurocore_missing_solver",
    )

    result = orchestrator.prove()

    assert result.passed is False
    assert result.exit_code == 127
    assert result.unavailable == ("sc_neurocore_missing_sby", "sc_neurocore_missing_solver")
    assert "missing formal dependency" in result.stderr_tail


@pytest.mark.skipif(
    shutil.which("sby") is None or shutil.which("cvc5") is None,
    reason="SymbiYosys/cvc5 external formal dependencies are unavailable",
)
def test_dense_layer_latency_example_proves_with_symbiyosys(tmp_path: Path) -> None:
    """The dense-layer latency example proves under SymbiYosys + cvc5."""
    result = TimingProofOrchestrator(EXAMPLE_SBY, temp_root=tmp_path).prove(timeout_s=120)

    assert result.passed, result.stdout_tail + "\n" + result.stderr_tail
    assert result.exit_code == 0
    assert result.unavailable == ()


def test_cdc_template_example_binds_property_in_open_source_subset() -> None:
    """The CDC example binds the template in the open-source assertion subset."""
    # The consumable surface MIF reads: the macro binding plus an open-source-flow
    # SymbiYosys task (procedural-immediate assertions, no concurrent SVA).
    sv_text = CDC_SV.read_text(encoding="utf-8")
    sby_text = CDC_SBY.read_text(encoding="utf-8")
    svh_text = ASSERTIONS_SVH.read_text(encoding="utf-8")

    assert "`SC_ASSERT_CDC_TWO_FLOP(aer_ingress" in sv_text
    assert "two_flop_synchroniser" in sv_text
    assert "`define SC_ASSERT_CDC_TWO_FLOP" in svh_text
    assert "sc_cdc_two_flop_monitor" in svh_text
    assert "mode bmc" in sby_text
    assert "smtbmc" in sby_text
    assert "read_verilog -formal" in sby_text
    # No concurrent SVA in the consumable surface (Yosys cannot prove it).
    assert "##[" not in svh_text


@pytest.mark.skipif(_FORMAL_UNAVAILABLE, reason="SymbiYosys/cvc5 unavailable")
def test_cdc_two_flop_synchroniser_proves_with_symbiyosys(tmp_path: Path) -> None:
    """A correct two-flop synchroniser proves under SymbiYosys + cvc5."""
    result = TimingProofOrchestrator(CDC_SBY, temp_root=tmp_path).prove(timeout_s=180)

    assert result.passed, result.stdout_tail + "\n" + result.stderr_tail
    assert result.exit_code == 0
    assert result.unavailable == ()


@pytest.mark.skipif(_FORMAL_UNAVAILABLE, reason="SymbiYosys/cvc5 unavailable")
def test_cdc_template_rejects_single_flop_synchroniser(tmp_path: Path) -> None:
    """The depth-2 CDC property rejects a one-flop synchroniser (non-vacuous)."""
    # A one-flop "synchroniser" delays the source by one cycle, not two; the depth-2
    # binding must catch it, proving the property is not vacuous.
    broken_sv = tmp_path / "broken_one_flop.sv"
    broken_sv.write_text(
        "`timescale 1ns/1ps\n"
        "`default_nettype none\n"
        '`include "timing_assertions.svh"\n'
        "module broken_one_flop_synchroniser (\n"
        "    input wire clk, input wire rst_n, input wire async_src\n"
        ");\n"
        "    reg past_valid = 1'b0;\n"
        "    reg sync_out = 1'b0;\n"
        "    wire meta_q = sync_out;\n"
        "    always @(posedge clk) begin\n"
        "        past_valid <= 1'b1;\n"
        "        if (!past_valid) assume (!rst_n); else assume (rst_n);\n"
        "    end\n"
        "    always @(posedge clk or negedge rst_n) begin\n"
        "        if (!rst_n) sync_out <= 1'b0; else sync_out <= async_src;\n"
        "    end\n"
        "    `SC_ASSERT_CDC_TWO_FLOP(aer_ingress, clk, rst_n, async_src, meta_q, sync_out, 2)\n"
        "endmodule\n"
        "`default_nettype wire\n",
        encoding="utf-8",
    )
    broken_sby = tmp_path / "broken_one_flop.sby"
    broken_sby.write_text(
        "[options]\nmode bmc\ndepth 12\n\n"
        "[engines]\nsmtbmc cvc5\n\n"
        "[script]\n"
        "read_verilog -formal -sv -I. timing_wrapper_lib.sv broken_one_flop.sv\n"
        "prep -top broken_one_flop_synchroniser\n\n"
        "[files]\n"
        f"{WRAPPER_LIB}\n{ASSERTIONS_SVH}\n{broken_sv}\n",
        encoding="utf-8",
    )

    result = TimingProofOrchestrator(broken_sby, temp_root=tmp_path).prove(timeout_s=180)

    assert result.passed is False, "depth-2 CDC property vacuously accepted a one-flop synchroniser"
    assert result.unavailable == ()
