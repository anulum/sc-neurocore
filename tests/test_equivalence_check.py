# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the machine-checked equivalence runner

"""Tests for the SymbiYosys equivalence runner.

The proof tests skip when ``sby`` / ``yosys`` are absent (as on CI without the
formal toolchain), mirroring the co-simulation tests' toolchain guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.compiler import _sby_runner, equivalence_check
from sc_neurocore.compiler.equivalence_check import (
    EquivalenceResult,
    formal_tools_available,
    prove_equivalence,
)
from sc_neurocore.compiler.equivalence_miter import MiterPort

_HAS_FORMAL = formal_tools_available()
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Two behaviourally identical registered adders (a+b vs b+a) — cheap to prove.
_TINY_DUT = """
module tiny_dut #(parameter integer W = 8)(
    input wire clk,
    input wire rst_n,
    input wire [W-1:0] a,
    input wire [W-1:0] b,
    output reg [W-1:0] y
);
    always @(posedge clk or negedge rst_n)
        if (!rst_n) y <= 0; else y <= a + b;
endmodule
"""

_TINY_REF = """
module tiny_ref #(parameter integer W = 8)(
    input wire clk,
    input wire rst_n,
    input wire [W-1:0] a,
    input wire [W-1:0] b,
    output reg [W-1:0] y
);
    wire [W-1:0] s = b + a;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) y <= 0; else y <= s;
endmodule
"""

# A deliberately wrong reference (off by one) — must be disproved.
_TINY_REF_BAD = _TINY_REF.replace("wire [W-1:0] s = b + a;", "wire [W-1:0] s = b + a + 1;")

_TINY_PORTS = [
    MiterPort("clk", 1, False, "input"),
    MiterPort("rst_n", 1, False, "input"),
    MiterPort("a", 8, False, "input"),
    MiterPort("b", 8, False, "input"),
    MiterPort("y", 8, False, "output"),
]

_needs_formal = pytest.mark.skipif(not _HAS_FORMAL, reason="SymbiYosys / Yosys not available")

# Quadratic integrate-and-fire (Lo et al. 2021): v <- max(V_MIN, v + (v*v >> K) + I);
# spike + reset when v >= V_THRESHOLD. A second neuron shape for the formal toolkit —
# a v*v SELF-multiply (the LIF only had state x free-input) and an inline ``wire = expr``.
# The golden reference and a structurally-distinct hardware DUT, proven equivalent.
_QIF_REF = """`timescale 1ns/1ps
module sc_qif_reference #(
    parameter integer DATA_WIDTH = 16, parameter integer K_SHIFT = 6,
    parameter signed [DATA_WIDTH-1:0] V_THRESHOLD = 1024,
    parameter signed [DATA_WIDTH-1:0] V_RESET = -1024,
    parameter signed [DATA_WIDTH-1:0] V_MIN = -2048
)(
    input wire clk, input wire rst_n, input wire signed [DATA_WIDTH-1:0] I_t,
    output reg spike_out, output reg signed [DATA_WIDTH-1:0] v_out
);
    reg signed [DATA_WIDTH-1:0] v;
    wire signed [2*DATA_WIDTH-1:0] v_sq = v * v;
    wire signed [DATA_WIDTH-1:0] dv = v_sq >>> K_SHIFT;
    wire signed [DATA_WIDTH-1:0] v_tmp = v + dv + I_t;
    wire signed [DATA_WIDTH-1:0] v_clamped = (v_tmp < V_MIN) ? V_MIN : v_tmp;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin v <= 0; v_out <= 0; spike_out <= 1'b0; end
        else if (v_clamped >= V_THRESHOLD) begin spike_out <= 1'b1; v <= V_RESET; v_out <= V_RESET; end
        else begin spike_out <= 1'b0; v <= v_clamped; v_out <= v_clamped; end
    end
endmodule
"""
_QIF_DUT = """`timescale 1ns/1ps
module sc_qif_dut #(
    parameter integer DATA_WIDTH = 16, parameter integer K_SHIFT = 6,
    parameter signed [DATA_WIDTH-1:0] V_THRESHOLD = 1024,
    parameter signed [DATA_WIDTH-1:0] V_RESET = -1024,
    parameter signed [DATA_WIDTH-1:0] V_MIN = -2048
)(
    input wire clk, input wire rst_n, input wire signed [DATA_WIDTH-1:0] I_t,
    output reg spike_out, output reg signed [DATA_WIDTH-1:0] v_out
);
    reg signed [DATA_WIDTH-1:0] v_reg;
    wire signed [2*DATA_WIDTH-1:0] v_squared = v_reg * v_reg;
    wire signed [DATA_WIDTH-1:0] quad_term = v_squared >>> K_SHIFT;
    wire signed [DATA_WIDTH-1:0] v_int = (v_reg + quad_term) + I_t;
    wire signed [DATA_WIDTH-1:0] v_floor = (v_int >= V_MIN) ? v_int : V_MIN;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin v_reg <= 0; v_out <= 0; spike_out <= 1'b0; end
        else if (v_floor >= V_THRESHOLD) begin spike_out <= 1'b1; v_reg <= V_RESET; v_out <= V_RESET; end
        else begin spike_out <= 1'b0; v_reg <= v_floor; v_out <= v_floor; end
    end
endmodule
"""


class TestFormalToolsAvailable:
    """Availability probe."""

    def test_returns_bool(self) -> None:
        assert isinstance(formal_tools_available(), bool)

    def test_false_when_sby_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            _sby_runner.shutil, "which", lambda name: None if name == "sby" else "/usr/bin/x"
        )
        assert formal_tools_available() is False

    def test_false_when_solver_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # sby + yosys present but the SMT solver absent (the CI-image case).
        monkeypatch.setattr(
            _sby_runner.shutil,
            "which",
            lambda name: None if name == "z3" else "/usr/bin/x",
        )
        assert formal_tools_available("z3") is False
        assert formal_tools_available("boolector") is True

    def test_raises_when_tools_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_sby_runner.shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError, match="must be on PATH"):
            prove_equivalence(
                _TINY_DUT,
                _TINY_REF,
                _TINY_PORTS,
                dut_top="tiny_dut",
                ref_top="tiny_ref",
            )


class TestSbyGeneration:
    """The generated .sby script (no run required)."""

    def test_sby_reads_all_sources_and_sets_mode_depth(self) -> None:
        sby = equivalence_check._generate_sby(
            "equiv_miter",
            ["equiv_miter.v", "tiny_ref.v", "tiny_dut.v"],
            depth=12,
            mode="bmc",
            engine="z3",
        )
        assert "mode bmc" in sby
        assert "bmc: depth 12" in sby
        assert "smtbmc z3" in sby
        assert "read -formal equiv_miter.v" in sby
        assert "prep -top equiv_miter" in sby

    def test_verdict_parsing_picks_last_done_line(self) -> None:
        stdout = "DONE (ERROR, rc=16)\nDONE (PASS, rc=0)\n"
        assert equivalence_check._parse_verdict(stdout) == ("PASS", 0)

    def test_verdict_parsing_unknown_when_absent(self) -> None:
        assert equivalence_check._parse_verdict("no summary here") == ("UNKNOWN", -1)


class TestRunnerErrors:
    """Error handling that does not require a real solver run."""

    def test_timeout_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import subprocess

        monkeypatch.setattr(_sby_runner.shutil, "which", lambda name: "/usr/bin/x")

        def _raise_timeout(*args: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd="sby", timeout=1.0)

        monkeypatch.setattr(_sby_runner.subprocess, "run", _raise_timeout)
        with pytest.raises(RuntimeError, match="timed out"):
            prove_equivalence(
                _TINY_DUT,
                _TINY_REF,
                _TINY_PORTS,
                dut_top="tiny_dut",
                ref_top="tiny_ref",
                timeout_s=1.0,
                workdir=tmp_path,
            )


class TestVerdictMapping:
    """Map a raw ``sby`` run onto an :class:`EquivalenceResult` without a solver."""

    def _patch(self, monkeypatch: pytest.MonkeyPatch, run: object) -> None:
        monkeypatch.setattr(equivalence_check, "formal_tools_available", lambda engine="z3": True)
        monkeypatch.setattr(equivalence_check, "run_sby_task", lambda *a, **k: run)

    def test_pass_maps_to_proven(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from sc_neurocore.compiler._sby_runner import SbyRun

        self._patch(
            monkeypatch, SbyRun(verdict="PASS", rc=0, returncode=0, summary=["summary: ok"])
        )
        result = prove_equivalence(
            _TINY_DUT,
            _TINY_REF,
            _TINY_PORTS,
            dut_top="tiny_dut",
            ref_top="tiny_ref",
            workdir=tmp_path,
        )
        assert result.proven is True
        assert result.verdict == "PASS"
        assert result.summary == ["summary: ok"]

    def test_fail_maps_to_disproven(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from sc_neurocore.compiler._sby_runner import SbyRun

        self._patch(
            monkeypatch,
            SbyRun(
                verdict="FAIL",
                rc=2,
                returncode=2,
                counterexample="failed assertion",
                trace_path=str(tmp_path / "t.vcd"),
            ),
        )
        result = prove_equivalence(
            _TINY_DUT,
            _TINY_REF_BAD,
            _TINY_PORTS,
            dut_top="tiny_dut",
            ref_top="tiny_ref",
            workdir=tmp_path,
        )
        assert result.proven is False
        assert result.counterexample == "failed assertion"
        assert result.trace_path == str(tmp_path / "t.vcd")

    def test_fail_without_counterexample_gets_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from sc_neurocore.compiler._sby_runner import SbyRun

        self._patch(monkeypatch, SbyRun(verdict="FAIL", rc=2, returncode=2))
        result = prove_equivalence(
            _TINY_DUT,
            _TINY_REF_BAD,
            _TINY_PORTS,
            dut_top="tiny_dut",
            ref_top="tiny_ref",
            workdir=tmp_path,
        )
        assert result.counterexample == "assertion failed"

    def test_incomplete_verdict_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from sc_neurocore.compiler._sby_runner import SbyRun

        self._patch(monkeypatch, SbyRun(verdict="ERROR", rc=16, returncode=16, stdout="boom"))
        with pytest.raises(RuntimeError, match="equivalence proof did not complete"):
            prove_equivalence(
                _TINY_DUT,
                _TINY_REF,
                _TINY_PORTS,
                dut_top="tiny_dut",
                ref_top="tiny_ref",
                workdir=tmp_path,
            )

    def test_inconclusive_kinduction_maps_to_unproven(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from sc_neurocore.compiler._sby_runner import SbyRun

        self._patch(monkeypatch, SbyRun(verdict="UNKNOWN", rc=4, returncode=4))
        result = prove_equivalence(
            _TINY_DUT,
            _TINY_REF,
            _TINY_PORTS,
            dut_top="tiny_dut",
            ref_top="tiny_ref",
            mode="prove",
            workdir=tmp_path,
        )
        assert result.proven is False
        assert result.verdict == "UNKNOWN"
        assert result.counterexample is None


@_needs_formal
class TestEquivalenceProof:
    """End-to-end machine-checked proofs (require the formal toolchain)."""

    def test_equivalent_modules_are_proven(self, tmp_path: Path) -> None:
        result = prove_equivalence(
            _TINY_DUT,
            _TINY_REF,
            _TINY_PORTS,
            dut_top="tiny_dut",
            ref_top="tiny_ref",
            depth=10,
            workdir=tmp_path,
        )
        assert isinstance(result, EquivalenceResult)
        assert result.proven is True
        assert result.verdict == "PASS"
        assert result.returncode == 0
        assert result.counterexample is None

    def test_inequivalent_modules_are_disproved(self, tmp_path: Path) -> None:
        result = prove_equivalence(
            _TINY_DUT,
            _TINY_REF_BAD,
            _TINY_PORTS,
            dut_top="tiny_dut",
            ref_top="tiny_ref",
            depth=10,
            workdir=tmp_path,
        )
        assert result.proven is False
        assert result.verdict == "FAIL"
        assert result.counterexample is not None
        assert "failed assertion" in result.counterexample.lower()
        assert result.trace_path is not None
        assert Path(result.trace_path).exists()

    def test_malformed_verilog_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="did not complete"):
            prove_equivalence(
                "module tiny_dut; this is not verilog",
                _TINY_REF,
                _TINY_PORTS,
                dut_top="tiny_dut",
                ref_top="tiny_ref",
                depth=4,
                workdir=tmp_path,
            )

    def test_generated_lif_matches_reference_model(self, tmp_path: Path) -> None:
        """The compiled LIF RTL primitive is equivalent to the reference model."""
        dut_path = _REPO_ROOT / "hdl" / "sc_lif_neuron.v"
        ref_path = _REPO_ROOT / "hdl" / "equiv" / "sc_lif_reference.v"
        if not dut_path.exists() or not ref_path.exists():
            pytest.skip("committed LIF DUT / reference not present")
        from sc_neurocore.compiler.equivalence_miter import parse_module_interface

        ref_src = ref_path.read_text(encoding="utf-8")
        ports = parse_module_interface(ref_src, "sc_lif_reference", params={"DATA_WIDTH": 16})
        common = {"DATA_WIDTH": 16, "FRACTION": 8, "V_REST": 0, "V_RESET": 0, "V_THRESHOLD": 256}
        result = prove_equivalence(
            dut_path.read_text(encoding="utf-8"),
            ref_src,
            ports,
            dut_top="sc_lif_neuron",
            ref_top="sc_lif_reference",
            dut_params={**common, "REFRACTORY_PERIOD": 0},
            ref_params=common,
            depth=4,
            workdir=tmp_path,
        )
        assert result.proven is True

    def test_reference_threshold_mismatch_is_caught(self, tmp_path: Path) -> None:
        """A reference with the wrong threshold parameter must be disproved."""
        dut_path = _REPO_ROOT / "hdl" / "sc_lif_neuron.v"
        ref_path = _REPO_ROOT / "hdl" / "equiv" / "sc_lif_reference.v"
        if not dut_path.exists() or not ref_path.exists():
            pytest.skip("committed LIF DUT / reference not present")
        from sc_neurocore.compiler.equivalence_miter import parse_module_interface

        ref_src = ref_path.read_text(encoding="utf-8")
        ports = parse_module_interface(ref_src, "sc_lif_reference", params={"DATA_WIDTH": 16})
        dut_params = {
            "DATA_WIDTH": 16,
            "FRACTION": 8,
            "V_REST": 0,
            "V_RESET": 0,
            "V_THRESHOLD": 256,
            "REFRACTORY_PERIOD": 0,
        }
        ref_params = {
            "DATA_WIDTH": 16,
            "FRACTION": 8,
            "V_REST": 0,
            "V_RESET": 0,
            "V_THRESHOLD": 512,
        }
        result = prove_equivalence(
            dut_path.read_text(encoding="utf-8"),
            ref_src,
            ports,
            dut_top="sc_lif_neuron",
            ref_top="sc_lif_reference",
            dut_params=dut_params,
            ref_params=ref_params,
            depth=8,
            workdir=tmp_path,
        )
        assert result.proven is False
        assert result.verdict == "FAIL"

    def test_whitebox_taps_make_lif_provable_unbounded(self, tmp_path: Path) -> None:
        """Exposing internal state as taps lets k-induction prove the LIF unbounded.

        Naive k-induction on the miter is intractable (the fixed-point multiplier
        diverges from unreachable start states). Instrumenting both modules to
        expose the membrane register and refractory counter turns the miter's
        output-equality asserts into the state-matching invariant, which makes
        ``mode="prove"`` converge. A narrow 4-bit datapath keeps the multiplier
        tractable for the SMT solver.
        """
        dut_path = _REPO_ROOT / "hdl" / "sc_lif_neuron.v"
        ref_path = _REPO_ROOT / "hdl" / "equiv" / "sc_lif_reference.v"
        if not dut_path.exists() or not ref_path.exists():
            pytest.skip("committed LIF DUT / reference not present")
        from sc_neurocore.compiler.equivalence_miter import parse_module_interface
        from sc_neurocore.compiler.whitebox_taps import StateTap, expose_state_taps

        dut_wb = expose_state_taps(
            dut_path.read_text(encoding="utf-8"),
            top="sc_lif_neuron",
            taps=[
                StateTap("v_state", "v_reg", msb="DATA_WIDTH-1", signed=True),
                StateTap("refr_state", "refractory_counter", msb="31"),
            ],
        )
        ref_wb = expose_state_taps(
            ref_path.read_text(encoding="utf-8"),
            top="sc_lif_reference",
            taps=[
                StateTap("v_state", "v", msb="DATA_WIDTH-1", signed=True),
                StateTap("refr_state", "32'd0", msb="31"),
            ],
        )
        common = {"DATA_WIDTH": 4, "FRACTION": 2, "V_REST": 0, "V_RESET": 0, "V_THRESHOLD": 4}
        ports = parse_module_interface(ref_wb, "sc_lif_reference", params={"DATA_WIDTH": 4})
        result = prove_equivalence(
            dut_wb,
            ref_wb,
            ports,
            dut_top="sc_lif_neuron",
            ref_top="sc_lif_reference",
            dut_params={**common, "REFRACTORY_PERIOD": 0},
            ref_params=common,
            mode="prove",
            depth=4,
            workdir=tmp_path,
        )
        assert result.proven is True
        assert result.verdict == "PASS"
        assert result.mode == "prove"

    def test_multiplier_abstraction_proves_lif_unbounded_full_width(self, tmp_path: Path) -> None:
        """Abstracting the multipliers lets k-induction prove the LIF at full width.

        Whitebox taps alone make k-induction *converge*, but bit-blasting the
        16-bit fixed-point multiplier keeps it intractable for the SMT solver.
        Lifting each product to a shared free input removes the multiplier from
        the solver entirely (the two instances see the same free product, so the
        abstraction is sound for a PASS), and the full 16-bit LIF proves unbounded.
        """
        dut_path = _REPO_ROOT / "hdl" / "sc_lif_neuron.v"
        ref_path = _REPO_ROOT / "hdl" / "equiv" / "sc_lif_reference.v"
        if not dut_path.exists() or not ref_path.exists():
            pytest.skip("committed LIF DUT / reference not present")
        from sc_neurocore.compiler.equivalence_miter import parse_module_interface
        from sc_neurocore.compiler.operator_abstraction import (
            LiftedSignal,
            abstract_to_free_inputs,
        )
        from sc_neurocore.compiler.whitebox_taps import StateTap, expose_state_taps

        dut = abstract_to_free_inputs(
            dut_path.read_text(encoding="utf-8"),
            top="sc_lif_neuron",
            signals=[
                LiftedSignal("leak_mul", "leak_product", msb="2*DATA_WIDTH-1", signed=True),
                LiftedSignal("in_mul", "input_product", msb="2*DATA_WIDTH-1", signed=True),
            ],
        )
        ref = abstract_to_free_inputs(
            ref_path.read_text(encoding="utf-8"),
            top="sc_lif_reference",
            signals=[
                LiftedSignal("leak_product", "leak_product", msb="2*DATA_WIDTH-1", signed=True),
                LiftedSignal("input_product", "input_product", msb="2*DATA_WIDTH-1", signed=True),
            ],
        )
        dut = expose_state_taps(
            dut,
            top="sc_lif_neuron",
            taps=[
                StateTap("v_state", "v_reg", msb="DATA_WIDTH-1", signed=True),
                StateTap("refr_state", "refractory_counter", msb="31"),
            ],
        )
        ref = expose_state_taps(
            ref,
            top="sc_lif_reference",
            taps=[
                StateTap("v_state", "v", msb="DATA_WIDTH-1", signed=True),
                StateTap("refr_state", "32'd0", msb="31"),
            ],
        )
        common = {"DATA_WIDTH": 16, "FRACTION": 8, "V_REST": 0, "V_RESET": 0, "V_THRESHOLD": 256}
        ports = parse_module_interface(ref, "sc_lif_reference", params={"DATA_WIDTH": 16})
        result = prove_equivalence(
            dut,
            ref,
            ports,
            dut_top="sc_lif_neuron",
            ref_top="sc_lif_reference",
            dut_params={**common, "REFRACTORY_PERIOD": 0},
            ref_params=common,
            mode="prove",
            depth=6,
            workdir=tmp_path,
        )
        assert result.proven is True
        assert result.verdict == "PASS"
        assert result.mode == "prove"

    def test_toolkit_generalises_to_quadratic_qif_unbounded(self, tmp_path: Path) -> None:
        """The whitebox-tap + multiplier-abstraction flow generalises to the QIF.

        A quadratic integrate-and-fire is a second neuron shape: its state update
        contains a ``v*v`` self-multiply (the LIF only multiplied state by a free
        input) declared inline as ``wire = expr``. Abstracting that product to a
        shared free input and tapping the single membrane state proves the
        structurally-distinct DUT and golden reference equivalent unbounded.
        """
        from sc_neurocore.compiler.equivalence_miter import parse_module_interface
        from sc_neurocore.compiler.operator_abstraction import (
            LiftedSignal,
            abstract_to_free_inputs,
        )
        from sc_neurocore.compiler.whitebox_taps import StateTap, expose_state_taps

        dut = abstract_to_free_inputs(
            _QIF_DUT,
            top="sc_qif_dut",
            signals=[LiftedSignal("v_squared", "v_sq_in", msb="2*DATA_WIDTH-1", signed=True)],
        )
        ref = abstract_to_free_inputs(
            _QIF_REF,
            top="sc_qif_reference",
            signals=[LiftedSignal("v_sq", "v_sq_in", msb="2*DATA_WIDTH-1", signed=True)],
        )
        dut = expose_state_taps(
            dut,
            top="sc_qif_dut",
            taps=[StateTap("v_state", "v_reg", msb="DATA_WIDTH-1", signed=True)],
        )
        ref = expose_state_taps(
            ref,
            top="sc_qif_reference",
            taps=[StateTap("v_state", "v", msb="DATA_WIDTH-1", signed=True)],
        )
        common = {
            "DATA_WIDTH": 16,
            "K_SHIFT": 6,
            "V_THRESHOLD": 1024,
            "V_RESET": -1024,
            "V_MIN": -2048,
        }
        ports = parse_module_interface(ref, "sc_qif_reference", params={"DATA_WIDTH": 16})
        result = prove_equivalence(
            dut,
            ref,
            ports,
            dut_top="sc_qif_dut",
            ref_top="sc_qif_reference",
            dut_params=common,
            ref_params=common,
            mode="prove",
            depth=6,
            workdir=tmp_path,
        )
        assert result.proven is True
        assert result.verdict == "PASS"
        assert result.mode == "prove"
