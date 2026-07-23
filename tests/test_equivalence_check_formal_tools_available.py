# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFormalToolsAvailable from former test_equivalence_check.py

"""Focused suite: TestFormalToolsAvailable from former test_equivalence_check.py."""

from __future__ import annotations

from tests.equivalence_check_support import *  # noqa: F403

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
