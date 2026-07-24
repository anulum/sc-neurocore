# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEmitterState from former test_c_expr_emitter.py

"""Focused suite: TestEmitterState from former test_c_expr_emitter.py."""

from __future__ import annotations

from tests.c_expr_emitter_support import *  # noqa: F403


class TestEmitterState:
    def test_free_vars_attribute(self) -> None:
        e = CExprEmitter({"v"})
        e.visit(__import__("ast").parse("v + leak", mode="eval").body)
        assert e.free_vars == ["leak"]
