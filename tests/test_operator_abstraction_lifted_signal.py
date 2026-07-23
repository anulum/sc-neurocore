# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLiftedSignal from former test_operator_abstraction.py

"""Focused suite: TestLiftedSignal from former test_operator_abstraction.py."""

from __future__ import annotations

from tests.operator_abstraction_support import *  # noqa: F403

class TestLiftedSignal:
    """The lifted-signal value type."""

    def test_declaration_sized_signed(self) -> None:
        sig = LiftedSignal("prod", "prod_in", msb="2*W-1", signed=True)
        assert sig.declaration() == "input wire signed [2*W-1:0] prod_in"

    def test_declaration_scalar_unsigned(self) -> None:
        assert LiftedSignal("q", "q_in").declaration() == "input wire q_in"
