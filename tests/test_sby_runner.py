# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the shared SymbiYosys task runner

"""Tests for the shared ``sby`` task runner.

The pure tests drive the parser, the tool probe, the verdict-completeness guard,
and the subprocess boundary with a crafted fake process, so they run everywhere.
One end-to-end test runs a real ``sby`` task and self-skips when the formal
toolchain (``sby`` / ``yosys`` / a solver) is absent, as on CI.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sc_neurocore.compiler import _sby_runner
from sc_neurocore.compiler._sby_runner import (
    SbyRun,
    formal_tools_available,
    is_inconclusive,
    parse_verdict,
    raise_for_incomplete,
    run_sby_task,
)

_HAS_FORMAL = formal_tools_available()
_needs_formal = pytest.mark.skipif(
    not _HAS_FORMAL, reason="SymbiYosys / Yosys / solver not available"
)


class _FakeProc:
    """Stand-in for a finished ``subprocess.run`` result."""

    def __init__(self, stdout: str, returncode: int) -> None:
        self.stdout = stdout
        self.returncode = returncode


class TestFormalToolsAvailable:
    """The toolchain probe."""

    def test_returns_bool(self) -> None:
        assert isinstance(formal_tools_available(), bool)

    def test_true_when_all_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_sby_runner.shutil, "which", lambda name: "/usr/bin/x")
        assert formal_tools_available("z3") is True

    def test_false_when_sby_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            _sby_runner.shutil, "which", lambda name: None if name == "sby" else "/usr/bin/x"
        )
        assert formal_tools_available() is False

    def test_false_when_yosys_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            _sby_runner.shutil, "which", lambda name: None if name == "yosys" else "/usr/bin/x"
        )
        assert formal_tools_available() is False

    def test_false_when_solver_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            _sby_runner.shutil, "which", lambda name: None if name == "z3" else "/usr/bin/x"
        )
        assert formal_tools_available("z3") is False
        assert formal_tools_available("boolector") is True


class TestParseVerdict:
    """Verdict extraction from ``sby`` output."""

    def test_picks_last_done_line(self) -> None:
        assert parse_verdict("DONE (ERROR, rc=16)\nDONE (PASS, rc=0)\n") == ("PASS", 0)

    def test_unknown_when_absent(self) -> None:
        assert parse_verdict("no summary here") == ("UNKNOWN", -1)

    def test_fail_verdict_with_code(self) -> None:
        assert parse_verdict("DONE (FAIL, rc=2)") == ("FAIL", 2)


class TestIsInconclusive:
    """The inconclusive k-induction signature (UNKNOWN, rc == 4)."""

    def test_unknown_rc4_is_inconclusive(self) -> None:
        assert is_inconclusive(SbyRun(verdict="UNKNOWN", rc=4, returncode=4)) is True

    def test_pass_is_not_inconclusive(self) -> None:
        assert is_inconclusive(SbyRun(verdict="PASS", rc=0, returncode=0)) is False

    def test_error_is_not_inconclusive(self) -> None:
        assert is_inconclusive(SbyRun(verdict="ERROR", rc=16, returncode=16)) is False

    def test_unknown_without_rc4_is_not_inconclusive(self) -> None:
        # A crashed run with no DONE line parses to (UNKNOWN, -1) — a tool failure,
        # not an inconclusive proof.
        assert is_inconclusive(SbyRun(verdict="UNKNOWN", rc=-1, returncode=1)) is False


class TestRaiseForIncomplete:
    """The verdict-completeness guard."""

    def test_pass_does_not_raise(self) -> None:
        raise_for_incomplete(SbyRun(verdict="PASS", rc=0, returncode=0), what="equivalence proof")

    def test_fail_does_not_raise(self) -> None:
        raise_for_incomplete(SbyRun(verdict="FAIL", rc=2, returncode=2), what="property proof")

    def test_inconclusive_does_not_raise(self) -> None:
        # A base-case-passed / induction-inconclusive k-induction is a real outcome.
        raise_for_incomplete(SbyRun(verdict="UNKNOWN", rc=4, returncode=4), what="property proof")

    def test_error_raises_with_label_and_tail(self) -> None:
        run = SbyRun(verdict="ERROR", rc=16, returncode=16, stdout="line1\nboom\n")
        with pytest.raises(RuntimeError, match="property proof did not complete"):
            raise_for_incomplete(run, what="property proof")

    def test_crash_without_done_line_raises(self) -> None:
        with pytest.raises(RuntimeError, match="verdict=UNKNOWN"):
            raise_for_incomplete(SbyRun(verdict="UNKNOWN", rc=-1, returncode=1), what="proof")


class TestRunSbyTask:
    """The subprocess boundary, exercised with a fake process."""

    def test_timeout_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def _raise_timeout(*args: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd="sby", timeout=1.0)

        monkeypatch.setattr(_sby_runner.subprocess, "run", _raise_timeout)
        with pytest.raises(RuntimeError, match="timed out after 1.0s"):
            run_sby_task(tmp_path, "x.sby", timeout_s=1.0)

    def test_pass_run_has_no_counterexample(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        stdout = "SBY summary: engine_0\nDONE (PASS, rc=0)\n"
        monkeypatch.setattr(_sby_runner.subprocess, "run", lambda *a, **k: _FakeProc(stdout, 0))
        run = run_sby_task(tmp_path, "x.sby", timeout_s=5.0)
        assert run.verdict == "PASS"
        assert run.rc == 0
        assert run.returncode == 0
        assert run.counterexample is None
        assert run.trace_path is None
        assert run.summary == ["SBY summary: engine_0"]

    def test_fail_run_extracts_counterexample_and_trace(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        stdout = (
            "summary: counterexample trace: bad_bmc/engine_0/trace.vcd\n"
            "summary: failed assertion mon.sva_i at mon_sva.sv:3 step 6\n"
            "DONE (FAIL, rc=2)\n"
        )
        monkeypatch.setattr(_sby_runner.subprocess, "run", lambda *a, **k: _FakeProc(stdout, 2))
        run = run_sby_task(tmp_path, "x.sby", timeout_s=5.0)
        assert run.verdict == "FAIL"
        assert run.counterexample is not None
        assert "failed assertion" in run.counterexample.lower()
        assert run.trace_path == str(tmp_path / "bad_bmc/engine_0/trace.vcd")

    def test_none_stdout_is_tolerated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            _sby_runner.subprocess,
            "run",
            lambda *a, **k: _FakeProc(None, 1),  # type: ignore[arg-type]
        )
        run = run_sby_task(tmp_path, "x.sby", timeout_s=5.0)
        assert run.verdict == "UNKNOWN"
        assert run.stdout == ""


@_needs_formal
class TestRunSbyTaskEndToEnd:
    """A real ``sby`` task with the toolchain present."""

    def test_trivial_true_assertion_passes(self, tmp_path: Path) -> None:
        (tmp_path / "m.v").write_text(
            "module m(input wire clk);\n"
            "  reg [3:0] c = 0;\n"
            "  always @(posedge clk) begin c <= c + 1; assert (c <= 15); end\n"
            "endmodule\n",
            encoding="utf-8",
        )
        (tmp_path / "m.sby").write_text(
            "[tasks]\nbmc\n[options]\nbmc: mode bmc\nbmc: depth 6\n"
            "[engines]\nsmtbmc z3\n[script]\nread -formal m.v\nprep -top m\n[files]\nm.v\n",
            encoding="utf-8",
        )
        run = run_sby_task(tmp_path, "m.sby", timeout_s=60.0)
        assert run.verdict == "PASS"
        assert run.returncode == 0
