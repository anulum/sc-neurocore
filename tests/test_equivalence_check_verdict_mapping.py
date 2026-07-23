# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVerdictMapping from former test_equivalence_check.py

"""Focused suite: TestVerdictMapping from former test_equivalence_check.py."""

from __future__ import annotations

from tests.equivalence_check_support import *  # noqa: F403

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
