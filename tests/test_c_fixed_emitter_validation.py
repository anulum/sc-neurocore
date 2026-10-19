# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestValidation from former test_c_fixed_emitter.py

"""Focused suite: TestValidation from former test_c_fixed_emitter.py."""

from __future__ import annotations

from tests.c_fixed_emitter_support import *  # noqa: F403


class TestValidation:
    """Configuration and syntax the bit-exact expression emitter refuses or accepts."""

    def test_invalid_lang_raises(self) -> None:
        """Only 'c' and 'rust' are accepted languages."""
        with pytest.raises(ValueError, match="lang must be"):
            emit_c_fixed_expr("v", {"v": "s->v"}, {}, Q, lang="go")

    def test_word_too_wide_raises(self) -> None:
        """A word whose product needs more than 64 bits is refused."""
        with pytest.raises(ValueError, match="2\\*data_width <= 64"):
            emit_c_fixed_expr("v", {"v": "s->v"}, {}, Q88(data_width=40, fraction=8))

    def test_q16_16_supported(self) -> None:
        """Q16.16 products fit the 64-bit intermediate and are emitted."""
        expr, *_ = emit_c_fixed_expr(
            "a * b", {"a": "s->a", "b": "s->b"}, {}, Q88(data_width=32, fraction=16)
        )
        assert "fxmul(" in expr

    def test_generic_visit_rejects_unknown_node(self) -> None:
        """An AST node outside the grammar, such as a list, is refused."""
        import ast

        emitter = _CFixedExprEmitter({}, {}, Q)
        with pytest.raises(ValueError, match="Unsupported AST node"):
            emitter.generic_visit(ast.parse("[1]", mode="eval").body)
