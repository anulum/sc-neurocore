# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFormalToolsAvailable from former test_sby_runner.py

"""Focused suite: TestFormalToolsAvailable from former test_sby_runner.py."""

from __future__ import annotations

import shutil
import pytest
from sc_neurocore.compiler._sby_runner import (
    formal_tools_available,
)


class TestFormalToolsAvailable:
    """The toolchain probe."""

    def test_returns_bool(self) -> None:
        """The toolchain probe returns a boolean."""
        assert isinstance(formal_tools_available(), bool)

    def test_true_when_all_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The probe is true when every tool is present."""
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/x")
        assert formal_tools_available("z3") is True

    def test_false_when_sby_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The probe is false without sby."""
        monkeypatch.setattr(shutil, "which", lambda name: None if name == "sby" else "/usr/bin/x")
        assert formal_tools_available() is False

    def test_false_when_yosys_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The probe is false without Yosys."""
        monkeypatch.setattr(shutil, "which", lambda name: None if name == "yosys" else "/usr/bin/x")
        assert formal_tools_available() is False

    def test_false_when_solver_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The probe is false without the named solver."""
        monkeypatch.setattr(shutil, "which", lambda name: None if name == "z3" else "/usr/bin/x")
        assert formal_tools_available("z3") is False
        assert formal_tools_available("boolector") is True
