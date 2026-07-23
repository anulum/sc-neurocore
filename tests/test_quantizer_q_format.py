# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQFormat from former test_quantizer.py

"""Focused suite: TestQFormat from former test_quantizer.py."""

from __future__ import annotations

from tests.quantizer_support import *  # noqa: F403

class TestQFormat:
    def test_parse_q88(self):
        q = QFormat.from_string("Q8.8")
        assert q.integer_bits == 8
        assert q.fraction_bits == 8
        assert q.total_bits == 16
        assert q.scale == 256

    def test_parse_q4_12(self):
        q = QFormat.from_string("Q4.12")
        assert q.total_bits == 16
        assert q.scale == 4096

    def test_range_q88(self):
        q = QFormat.from_string("Q8.8")
        assert q.min_val == -128.0
        assert q.max_val == pytest.approx(127.99609375)
        assert q.min_value == q.min_val
        assert q.max_value == q.max_val
        assert q.q_label == "Q8.8"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Expected format"):
            QFormat.from_string("float32")

    def test_invalid_bit_contracts_raise(self):
        with pytest.raises(ValueError, match="sign bit"):
            QFormat(0, 8)
        with pytest.raises(ValueError, match="non-negative"):
            QFormat(8, -1)
        with pytest.raises(TypeError, match="integer_bits"):
            QFormat(True, 8)
