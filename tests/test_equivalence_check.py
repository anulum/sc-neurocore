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

from sc_neurocore.compiler import equivalence_check
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


class TestFormalToolsAvailable:
    """Availability probe."""

    def test_returns_bool(self) -> None:
        assert isinstance(formal_tools_available(), bool)

    def test_false_when_sby_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            equivalence_check.shutil, "which", lambda name: None if name == "sby" else "/usr/bin/x"
        )
        assert formal_tools_available() is False

    def test_raises_when_tools_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(equivalence_check.shutil, "which", lambda name: None)
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

        monkeypatch.setattr(equivalence_check.shutil, "which", lambda name: "/usr/bin/x")

        def _raise_timeout(*args: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd="sby", timeout=1.0)

        monkeypatch.setattr(equivalence_check.subprocess, "run", _raise_timeout)
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
