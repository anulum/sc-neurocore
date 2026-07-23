# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestUnsupportedNodes from former test_c_expr_emitter.py

"""Focused suite: TestUnsupportedNodes from former test_c_expr_emitter.py."""

from __future__ import annotations

from tests.c_expr_emitter_support import *  # noqa: F403

class TestUnsupportedNodes:
    def test_list_node_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported AST node"):
            _emit("[1, 2, 3]")

    def test_unsupported_binop_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported binary op"):
            _emit("a % b")
