# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSSAEnvironment from former test_compiler_export.py

"""Focused suite: TestSSAEnvironment from former test_compiler_export.py."""

from __future__ import annotations

from compiler_export_support import *  # noqa: F403


class TestSSAEnvironment(unittest.TestCase):
    """Verify SSA register allocation and external input lookup."""

    def test_allocate_sequential(self) -> None:
        ssa = SSAEnvironment()
        r0 = ssa.allocate("a")
        r1 = ssa.allocate("b")
        self.assertEqual(r0, "%0")
        self.assertEqual(r1, "%1")

    def test_get_allocated(self) -> None:
        ssa = SSAEnvironment()
        ssa.allocate("x")
        self.assertEqual(ssa.get("x"), "%0")

    def test_get_unallocated_returns_global(self) -> None:
        ssa = SSAEnvironment()
        self.assertEqual(ssa.get("input_a"), "%input_a")
