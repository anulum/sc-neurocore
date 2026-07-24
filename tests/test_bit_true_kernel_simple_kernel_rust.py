# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSimpleKernelRust from former test_bit_true_kernel.py

"""Focused suite: TestSimpleKernelRust from former test_bit_true_kernel.py."""

from __future__ import annotations

from tests.bit_true_kernel_support import *  # noqa: F403


class TestSimpleKernelRust:
    def test_substrings(self) -> None:
        code = generate_bittrue_kernel("sc_lif", {"v": "a + b"}, language="rust")
        for s in ("pub struct", "fn sat", "clamp"):
            assert s in code

    def test_step_computes(self) -> None:
        code = generate_bittrue_kernel("sc_lif", {"v": "a + b"}, language="rust")
        assert "let _next_v" in code and "self.v = _next_v;" in code

    def test_free_variables_become_arguments(self) -> None:
        code = generate_bittrue_kernel("sc_lif", {"v": "a + b"}, language="rust")
        assert ", a: i16" in code and ", b: i16" in code

    def test_positive_modulo_uses_floor_remainder_helper(self) -> None:
        code = generate_bittrue_kernel("sc_phase", {"v": "v % 2.0"}, language="rust")
        assert "fn fxmod(value: i64, period: i64) -> i16" in code
        assert "if remainder < 0 { remainder += period; }" in code
