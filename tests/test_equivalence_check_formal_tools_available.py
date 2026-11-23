# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFormalToolsAvailable from former test_equivalence_check.py

"""Focused suite: TestFormalToolsAvailable from former test_equivalence_check.py."""

from __future__ import annotations

import pytest
from sc_neurocore.compiler.equivalence_check import (
    formal_tools_available,
    prove_equivalence,
)
from tests.equivalence_check_support import (
    _TINY_DUT,
    _TINY_PORTS,
    _TINY_REF,
)


class TestFormalToolsAvailable:
    """Probe whether the formal toolchain and a chosen SMT solver are on PATH."""

    def test_returns_bool(self) -> None:
        """The tool probe answers with a boolean."""
        assert isinstance(formal_tools_available(), bool)

    def test_false_when_sby_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without sby on the path the formal tools are reported absent."""
        monkeypatch.setattr(
            "sc_neurocore.compiler._sby_runner.shutil.which",
            lambda name: None if name == "sby" else "/usr/bin/x",
        )
        assert formal_tools_available() is False

    def test_false_when_solver_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing SMT solver makes only that solver unavailable."""
        # sby + yosys present but the SMT solver absent (the CI-image case).
        monkeypatch.setattr(
            "sc_neurocore.compiler._sby_runner.shutil.which",
            lambda name: None if name == "z3" else "/usr/bin/x",
        )
        assert formal_tools_available("z3") is False
        assert formal_tools_available("boolector") is True

    def test_raises_when_tools_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A proof without the formal tools raises and names the PATH requirement."""
        monkeypatch.setattr("sc_neurocore.compiler._sby_runner.shutil.which", lambda name: None)
        with pytest.raises(RuntimeError, match="must be on PATH"):
            prove_equivalence(
                _TINY_DUT,
                _TINY_REF,
                _TINY_PORTS,
                dut_top="tiny_dut",
                ref_top="tiny_ref",
            )
