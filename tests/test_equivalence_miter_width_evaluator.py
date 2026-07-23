# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWidthEvaluator from former test_equivalence_miter.py

"""Focused suite: TestWidthEvaluator from former test_equivalence_miter.py."""

from __future__ import annotations

from tests.equivalence_miter_support import *  # noqa: F403

class TestWidthEvaluator:
    """The restricted arithmetic evaluator behind parameter-dependent widths."""

    @pytest.mark.parametrize(
        ("expr", "params", "expected"),
        [
            ("7", {}, 7),
            ("W - 1", {"W": 32}, 31),
            ("2 + 3", {}, 5),
            ("2 * 4", {}, 8),
            ("8 // 2", {}, 4),
            ("1 << 4", {}, 16),
            ("16 >> 2", {}, 4),
            ("- -8", {}, 8),
            ("W // 2 + 1", {"W": 16}, 9),
        ],
    )
    def test_operators(self, expr: str, params: dict[str, int], expected: int) -> None:
        assert _eval_width_expr(expr, params) == expected

    def test_non_integer_literal_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-integer literal"):
            _eval_width_expr("1.5", {})

    def test_unknown_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown parameter"):
            _eval_width_expr("MISSING", {})

    def test_unsupported_operator_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported width expression"):
            _eval_width_expr("2 ** 3", {})
