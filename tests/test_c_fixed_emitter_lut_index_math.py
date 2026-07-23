# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLutIndexMath from former test_c_fixed_emitter.py

"""Focused suite: TestLutIndexMath from former test_c_fixed_emitter.py."""

from __future__ import annotations

from tests.c_fixed_emitter_support import *  # noqa: F403

class TestLutIndexMath:
    def test_symmetric_offset_and_shift(self):
        # exp uses [-16,16) step 0.125 → offset 16<<8=4096, shift 8-3=5
        _e, stmts, *_ = _c("exp(v)", state={"v": "s->v"})
        raw = next(s for s in stmts if "_raw" in s)
        assert "4096" in raw and ">> 5" in raw

    def test_positive_log_offset_and_shift(self):
        # log uses [1/256, 8+1/256) step 1/32 → offset 1, shift 8-5=3
        _e, stmts, *_ = _c("log(v)", state={"v": "s->v"})
        raw = next(s for s in stmts if "_raw" in s)
        assert "- 1" in raw and ">> 3" in raw

    def test_lut_start_offsets_table_names(self):
        _e, _s, tables, _fv, count, _u = _c("exp(v)", state={"v": "s->v"}, lut_start=3)
        assert "_exp_lut3" in tables and count == 4
