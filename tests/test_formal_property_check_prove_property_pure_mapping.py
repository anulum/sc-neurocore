# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestProvePropertyPureMapping from former test_formal_property_check.py

"""Focused suite: TestProvePropertyPureMapping from former test_formal_property_check.py."""

from __future__ import annotations

from tests.formal_property_check_support import *  # noqa: F403


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

    def test_inconclusive_kinduction_maps_to_unproven_without_counterexample(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            formal_property_check, "formal_tools_available", lambda engine="z3": True
        )
        monkeypatch.setattr(
            formal_property_check,
            "run_sby_task",
            lambda *a, **k: _fake_run(verdict="UNKNOWN", rc=4, returncode=4),
        )
        result = prove_property(_ACC_RTL, _ACC_SVA_WEAK, top="acc", mode="prove", workdir=tmp_path)
        assert result.proven is False
        assert result.verdict == "UNKNOWN"
        assert result.counterexample is None
        assert result.trace_path is None
