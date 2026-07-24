# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestProvePropertyEndToEnd from former test_formal_property_check.py

"""Focused suite: TestProvePropertyEndToEnd from former test_formal_property_check.py."""

from __future__ import annotations

from tests.formal_property_check_support import *  # noqa: F403


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

    def test_kinduction_with_strengthening_proves_unbounded(self, tmp_path: Path) -> None:
        # The strengthening lemma makes the accumulator bound 1-inductive, so
        # k-induction proves it unboundedly at a small depth independent of length.
        result = prove_property(
            _ACC_RTL, _ACC_SVA_STRONG, top="acc", mode="prove", depth=8, workdir=tmp_path
        )
        assert result.proven is True
        assert result.verdict == "PASS"

    def test_kinduction_without_strengthening_is_inconclusive(self, tmp_path: Path) -> None:
        # The target bound alone is true but not inductive: base case holds, the
        # induction step does not converge -> inconclusive, not disproved.
        result = prove_property(
            _ACC_RTL, _ACC_SVA_WEAK, top="acc", mode="prove", depth=8, workdir=tmp_path
        )
        assert result.proven is False
        assert result.verdict == "UNKNOWN"
        assert result.counterexample is None
