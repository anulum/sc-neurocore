# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSimpleKernelC from former test_bit_true_kernel.py

"""Focused suite: TestSimpleKernelC from former test_bit_true_kernel.py."""

from __future__ import annotations

from tests.bit_true_kernel_support import *  # noqa: F403


class TestSimpleKernelC:
    def test_substrings(self) -> None:
        code = generate_bittrue_kernel("sc_lif", {"v": "a + b"})
        for s in ("#include <stdint.h>", "sc_lif_state_t", "sat(", "fxmul("):
            assert s in code

    def test_step_is_not_a_noop(self) -> None:
        code = generate_bittrue_kernel("sc_lif", {"v": "a + b"})
        assert "_next_v = sat(" in code
        assert "s->v = _next_v;" in code
        assert "/* update */" not in code  # the old placeholder is gone

    def test_multi_var_struct(self) -> None:
        code = generate_bittrue_kernel("sc_izh", {"v": "a * b", "u": "c + d"})
        assert "int16_t v;" in code and "int16_t u;" in code

    def test_free_variables_become_arguments(self) -> None:
        code = generate_bittrue_kernel("sc_lif", {"v": "a + b"})
        assert "int16_t a" in code and "int16_t b" in code

    def test_input_current_becomes_argument(self) -> None:
        code = generate_bittrue_kernel("sc_lif", {"v": "I - v"})
        assert "int16_t I_t" in code

    def test_transcendental_declares_table(self) -> None:
        code = generate_bittrue_kernel("sc_th", {"v": "tanh(v)"})
        assert "static const int16_t _tanh_lut0" in code

    def test_positive_modulo_uses_floor_remainder_helper(self) -> None:
        code = generate_bittrue_kernel("sc_phase", {"v": "v % 2.0"})
        assert "static inline int16_t fxmod" in code
        assert "if (remainder < 0) { remainder += period; }" in code
        assert "fxmod(" in code
