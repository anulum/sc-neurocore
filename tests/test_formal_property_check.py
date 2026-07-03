# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the machine-checked RTL property runner

"""Tests for the SymbiYosys RTL-property runner.

Pure tests drive the ``.sby`` generation and the verdict-to-result mapping with a
fake run, so every branch runs without the toolchain (as on CI). The end-to-end
proofs run real ``sby`` tasks and self-skip when the formal toolchain is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.compiler import formal_property_check
from sc_neurocore.compiler._sby_runner import SbyRun
from sc_neurocore.compiler.formal_property_check import (
    PropertyProofResult,
    formal_tools_available,
    prove_property,
)

_HAS_FORMAL = formal_tools_available()
_needs_formal = pytest.mark.skipif(
    not _HAS_FORMAL, reason="SymbiYosys / Yosys / solver not available"
)

# A 4-bit counter whose value provably never exceeds 15 (it wraps): PASS.
_CAP_RTL = """`timescale 1ns/1ps
module cap(input wire clk, output wire [3:0] c_o);
    reg [3:0] c = 4'd0;
    always @(posedge clk) c <= c + 4'd1;
    assign c_o = c;
`ifdef FORMAL
    cap_sva sva_i (.clk(clk), .c(c));
`endif
endmodule
"""
_CAP_SVA = """`timescale 1ns/1ps
module cap_sva(input logic clk, input logic [3:0] c);
    always @(posedge clk) assert (c <= 4'd15);
endmodule
"""

# The same counter asserting a bound it must eventually break: FAIL at step 5.
_BAD_RTL = _CAP_RTL.replace("cap", "bad")
_BAD_SVA = """`timescale 1ns/1ps
module bad_sva(input logic clk, input logic [3:0] c);
    always @(posedge clk) assert (c < 4'd5);
endmodule
"""


def _fake_run(**fields: object) -> SbyRun:
    base = {"verdict": "PASS", "rc": 0, "returncode": 0}
    base.update(fields)
    return SbyRun(**base)  # type: ignore[arg-type]


class TestPropertyProofResult:
    """The result dataclass."""

    def test_defaults(self) -> None:
        result = PropertyProofResult(
            proven=True, verdict="PASS", mode="bmc", depth=8, engine="z3", returncode=0
        )
        assert result.counterexample is None
        assert result.trace_path is None
        assert result.summary == []


class TestGeneratePropertySby:
    """The generated ``.sby`` script (no run required)."""

    def test_reads_rtl_then_sva_and_preps_top(self) -> None:
        sby = formal_property_check._generate_property_sby(
            "mon", "mon.v", "mon_sva.sv", depth=18, mode="bmc", engine="z3"
        )
        assert "mode bmc" in sby
        assert "bmc: depth 18" in sby
        assert "smtbmc z3" in sby
        assert sby.index("read -formal mon.v") < sby.index("read -sv -formal mon_sva.sv")
        assert "prep -top mon" in sby


class TestProvePropertyPureMapping:
    """Verdict-to-result mapping, exercised with a fake run."""

    def test_raises_when_tools_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            formal_property_check, "formal_tools_available", lambda engine="z3": False
        )
        with pytest.raises(RuntimeError, match="must be on PATH"):
            prove_property(_CAP_RTL, _CAP_SVA, top="cap")

    def test_pass_maps_to_proven(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            formal_property_check, "formal_tools_available", lambda engine="z3": True
        )
        monkeypatch.setattr(
            formal_property_check,
            "run_sby_task",
            lambda *a, **k: _fake_run(verdict="PASS", summary=["summary: ok"]),
        )
        result = prove_property(_CAP_RTL, _CAP_SVA, top="cap", depth=8, workdir=tmp_path)
        assert result.proven is True
        assert result.verdict == "PASS"
        assert result.mode == "bmc"
        assert result.depth == 8
        assert result.summary == ["summary: ok"]
        # The sources were materialised for the run.
        assert (tmp_path / "cap.v").is_file()
        assert (tmp_path / "cap_sva.sv").is_file()
        assert (tmp_path / "cap.sby").is_file()

    def test_fail_maps_to_disproven_with_counterexample(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            formal_property_check, "formal_tools_available", lambda engine="z3": True
        )
        monkeypatch.setattr(
            formal_property_check,
            "run_sby_task",
            lambda *a, **k: _fake_run(
                verdict="FAIL",
                rc=2,
                returncode=2,
                counterexample="failed assertion at step 5",
                trace_path=str(tmp_path / "trace.vcd"),
            ),
        )
        result = prove_property(_BAD_RTL, _BAD_SVA, top="bad", depth=8, workdir=tmp_path)
        assert result.proven is False
        assert result.verdict == "FAIL"
        assert result.counterexample == "failed assertion at step 5"
        assert result.trace_path == str(tmp_path / "trace.vcd")

    def test_fail_without_counterexample_text_gets_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            formal_property_check, "formal_tools_available", lambda engine="z3": True
        )
        monkeypatch.setattr(
            formal_property_check,
            "run_sby_task",
            lambda *a, **k: _fake_run(verdict="FAIL", rc=2, returncode=2),
        )
        result = prove_property(_BAD_RTL, _BAD_SVA, top="bad", workdir=tmp_path)
        assert result.counterexample == "assertion failed"

    def test_incomplete_verdict_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            formal_property_check, "formal_tools_available", lambda engine="z3": True
        )
        monkeypatch.setattr(
            formal_property_check,
            "run_sby_task",
            lambda *a, **k: _fake_run(verdict="ERROR", rc=16, returncode=16, stdout="boom"),
        )
        with pytest.raises(RuntimeError, match="property proof did not complete"):
            prove_property(_CAP_RTL, _CAP_SVA, top="cap", workdir=tmp_path)


@_needs_formal
class TestProvePropertyEndToEnd:
    """Real machine-checked property proofs (require the formal toolchain)."""

    def test_bounded_monitor_is_proven(self, tmp_path: Path) -> None:
        result = prove_property(_CAP_RTL, _CAP_SVA, top="cap", depth=8, workdir=tmp_path)
        assert isinstance(result, PropertyProofResult)
        assert result.proven is True
        assert result.verdict == "PASS"
        assert result.returncode == 0

    def test_violating_monitor_is_disproved(self, tmp_path: Path) -> None:
        result = prove_property(_BAD_RTL, _BAD_SVA, top="bad", depth=8, workdir=tmp_path)
        assert result.proven is False
        assert result.verdict == "FAIL"
        assert result.counterexample is not None
        assert result.trace_path is not None
        assert Path(result.trace_path).exists()

    def test_malformed_rtl_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="did not complete"):
            prove_property(
                "module cap; not verilog", _CAP_SVA, top="cap", depth=4, workdir=tmp_path
            )

    def test_default_workdir_is_created(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = prove_property(_CAP_RTL, _CAP_SVA, top="cap", depth=6)
        assert result.proven is True
        assert (tmp_path / "cap_property_work").is_dir()
