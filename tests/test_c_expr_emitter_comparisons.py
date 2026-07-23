# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestComparisons from former test_c_expr_emitter.py

"""Focused suite: TestComparisons from former test_c_expr_emitter.py."""

from __future__ import annotations

from tests.c_expr_emitter_support import *  # noqa: F403

class TestComparisons:
    def test_all_comparisons(self) -> None:
        assert _emit("v > 1.0", {"v"}) == "(v > fp_t(1.0))"
        assert _emit("v >= 1.0", {"v"}) == "(v >= fp_t(1.0))"
        assert _emit("v < 1.0", {"v"}) == "(v < fp_t(1.0))"
        assert _emit("v <= 1.0", {"v"}) == "(v <= fp_t(1.0))"

    def test_unsupported_comparison_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported comparison"):
            _emit("v == 1.0", {"v"})
