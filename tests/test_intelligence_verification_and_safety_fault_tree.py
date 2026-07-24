# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFaultTree from former test_intelligence_verification_and_safety.py

"""Focused suite: TestFaultTree from former test_intelligence_verification_and_safety.py."""

from __future__ import annotations

from tests.intelligence_verification_and_safety_support import *  # noqa: F403


class TestFaultTree:
    def test_basic(self):
        from sc_neurocore.compiler.intelligence import generate_fault_tree

        ft = generate_fault_tree("sc_lif", {"v": "a", "u": "b"})
        assert "SYSTEM_FAILURE" in ft.top_event
        assert len(ft.basic_events) >= 6  # 2 vars * 2 + 2 common
        assert len(ft.mcs) == len(ft.basic_events)

    def test_single_var(self):
        from sc_neurocore.compiler.intelligence import generate_fault_tree

        ft = generate_fault_tree("sc_lif", {"v": "a"})
        assert len(ft.basic_events) == 4  # 1 var * 2 + 2 common
