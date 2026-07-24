# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMZICompiler from former test_photonic_noc.py

"""Focused suite: TestMZICompiler from former test_photonic_noc.py."""

from __future__ import annotations

from photonic_noc_support import *  # noqa: F403


class TestMZICompiler:
    """MZI gate compilation tests."""

    def test_compile_and_gate(self) -> None:
        mzi = MZICompiler().compile_gate("AND", [0, 1], 2)
        assert mzi.operation == "AND"
        assert abs(mzi.phase_shift_rad - math.pi / 2) < 1e-10

    def test_compile_not_gate(self) -> None:
        mzi = MZICompiler().compile_gate("NOT", [0], 1)
        assert abs(mzi.phase_shift_rad - math.pi) < 1e-10

    def test_compile_network(self) -> None:
        gates = [
            {"type": "MUL", "inputs": [0, 1], "output": 2},
            {"type": "ADD", "inputs": [2, 3], "output": 4},
        ]
        mzis = MZICompiler().compile_network(gates)
        assert len(mzis) == 2
