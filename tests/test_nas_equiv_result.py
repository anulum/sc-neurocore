# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEquivResult from former test_nas.py

"""Focused suite: TestEquivResult from former test_nas.py."""

from __future__ import annotations

from tests.nas_support import *  # noqa: F403


class TestEquivResult:
    def test_summary_pass(self) -> None:
        r = EquivResult(module="sc_lif_neuron", passed=True, depth=30, engine="z3", log="ok")
        assert "PROVED" in r.summary()

    def test_summary_fail(self) -> None:
        r = EquivResult(module="sc_lif_neuron", passed=False, depth=30, engine="z3", log="err")
        assert "FAILED" in r.summary()
