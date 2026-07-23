# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCompare from former test_c_fixed_emitter.py

"""Focused suite: TestCompare from former test_c_fixed_emitter.py."""

from __future__ import annotations

from tests.c_fixed_emitter_support import *  # noqa: F403

class TestCompare:
    @pytest.mark.parametrize("op,sym", [(">", ">"), (">=", ">="), ("<", "<"), ("<=", "<=")])
    def test_comparisons(self, op, sym):
        expr, *_ = _c(f"v {op} 30.0", state={"v": "s->v"})
        assert sym in expr

    def test_chained_comparison(self):
        expr, *_ = _c("0.0 < v", state={"v": "s->v"})
        assert "<" in expr

    def test_unsupported_comparison_raises(self):
        with pytest.raises(ValueError, match="Unsupported comparison"):
            _c("v == 1.0", state={"v": "s->v"})
