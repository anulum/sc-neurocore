# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTypeHelpers from former test_bit_true_kernel.py

"""Focused suite: TestTypeHelpers from former test_bit_true_kernel.py."""

from __future__ import annotations

from tests.bit_true_kernel_support import *  # noqa: F403

class TestTypeHelpers:
    @pytest.mark.parametrize(
        "dw,expected", [(8, "int8_t"), (16, "int16_t"), (32, "int32_t"), (64, "int64_t")]
    )
    def test_ctype_native(self, dw: int, expected: str) -> None:
        assert _ctype(dw) == expected

    def test_ctype_non_native_widens(self) -> None:
        assert _ctype(24) == "int32_t"
        assert _ctype(48) == "int64_t"

    @pytest.mark.parametrize("dw,expected", [(8, "i8"), (16, "i16"), (32, "i32"), (64, "i64")])
    def test_rtype_native(self, dw: int, expected: str) -> None:
        assert _rtype(dw) == expected

    def test_rtype_non_native_widens(self) -> None:
        assert _rtype(24) == "i32"
        assert _rtype(48) == "i64"

    def test_accumulate_bias_saturate(self) -> None:
        assert _accumulate_bias("x", "saturate") == "sat(x)"

    def test_accumulate_bias_wrap(self) -> None:
        assert _accumulate_bias("x", "wrap") == "sc_wrap(x, WORD_BITS)"

    def test_format_tables_empty(self) -> None:
        assert _format_tables_c({}, 16) == []
