# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the shared expression-lowering tables

"""Tests for the target-independent expression LUT tables.

These bind the shared numerics that every lowering backend must agree on and
cross-check that the Verilog emitter still produces byte-identical values after
delegating to this module.
"""

from __future__ import annotations

import ast
import math

from sc_neurocore.compiler import expr_lut_tables as tables
from sc_neurocore.compiler.equation_compiler import Q88, _VerilogExprEmitter


class TestSupportedFunctions:
    def test_contains_transcendentals_and_helpers(self) -> None:
        assert {"exp", "log", "sqrt", "tanh", "cosh", "exprel", "sin", "cos"} <= (
            tables.SUPPORTED_FUNCTIONS
        )
        assert {"sigmoid", "expit", "abs", "clip", "max", "min"} <= tables.SUPPORTED_FUNCTIONS

    def test_is_frozen(self) -> None:
        assert isinstance(tables.SUPPORTED_FUNCTIONS, frozenset)


class TestConstFloat:
    def _fold(self, expr: str) -> float | None:
        return tables.const_float(ast.parse(expr, mode="eval").body)

    def test_literal(self) -> None:
        assert self._fold("3.5") == 3.5

    def test_unary_minus(self) -> None:
        assert self._fold("-2.0") == -2.0

    def test_division(self) -> None:
        assert self._fold("1.0 / 3.0") == 1.0 / 3.0

    def test_mult_add_sub(self) -> None:
        assert self._fold("2 * 3") == 6.0
        assert self._fold("2 + 3") == 5.0
        assert self._fold("5 - 3") == 2.0

    def test_division_by_zero_is_none(self) -> None:
        assert self._fold("1.0 / 0.0") is None

    def test_non_constant_is_none(self) -> None:
        assert self._fold("x + 1") is None

    def test_nested_non_constant_is_none(self) -> None:
        assert self._fold("x * 2 + 1") is None


class TestSamplePoints:
    def test_length_and_endpoints(self) -> None:
        pts = tables.symmetric_sample_points()
        assert len(pts) == 256
        assert pts[0] == -16.0
        assert pts[128] == 0.0
        assert pts[-1] == -16.0 + 255 * 0.125

    def test_deterministic(self) -> None:
        assert tables.symmetric_sample_points() == tables.symmetric_sample_points()


class TestLutEntries:
    def test_symmetric_luts_have_256_entries(self) -> None:
        assert len(tables.exp_lut_entries(16, 8)) == 256
        assert len(tables.tanh_lut_entries(8)) == 256
        assert len(tables.cosh_lut_entries(16, 8)) == 256
        assert len(tables.exprel_lut_entries(16, 8)) == 256
        assert len(tables.sigmoid_lut_entries(8)) == 256
        assert len(tables.sin_lut_entries(8)) == 256
        assert len(tables.cos_lut_entries(8)) == 256
        assert len(tables.cbrt_lut_entries(8)) == 256

    def test_short_luts_have_16_entries(self) -> None:
        assert len(tables.log_lut_entries(8)) == 16
        assert len(tables.sqrt_lut_entries(8)) == 16

    def test_exp_zero_point_and_saturation(self) -> None:
        exp = tables.exp_lut_entries(16, 8)
        # x = 0 at index 128 -> exp(0) * 256 = 256.
        assert exp[128] == 256
        # Large positive x saturates to the signed 16-bit max.
        assert exp[255] == (1 << 15) - 1

    def test_saturation_cap_tracks_width(self) -> None:
        # An 8-bit word saturates far lower than a 16-bit word.
        exp8 = tables.exp_lut_entries(8, 4)
        assert max(exp8) == (1 << 7) - 1

    def test_tanh_is_bounded(self) -> None:
        tanh = tables.tanh_lut_entries(8)
        assert max(tanh) <= (1 << 8)
        assert min(tanh) >= -(1 << 8)

    def test_exprel_limit_at_zero_is_one(self) -> None:
        # exprel(0) = 1 -> 1 * 256 = 256 at index 128.
        assert tables.exprel_lut_entries(16, 8)[128] == 256

    def test_all_entries_are_int(self) -> None:
        for entry in tables.exp_lut_entries(16, 8):
            assert isinstance(entry, int)
        for entry in tables.sqrt_lut_entries(8):
            assert isinstance(entry, int)

    def test_cbrt_is_odd_symmetric(self) -> None:
        cbrt = tables.cbrt_lut_entries(8)
        # index 128 is x=0; index 128+k and 128-k are negatives of each other.
        assert cbrt[128] == 0
        assert cbrt[128 + 40] == -cbrt[128 - 40]

    def test_sin_matches_reference(self) -> None:
        sin = tables.sin_lut_entries(8)
        pts = tables.symmetric_sample_points()
        assert sin[200] == int(round(math.sin(pts[200]) * 256))


class TestEmitterDelegationParity:
    """The Verilog emitter must return byte-identical tables after delegation."""

    def _emitter(self, data_width: int = 16, fraction: int = 8) -> _VerilogExprEmitter:
        return _VerilogExprEmitter(set(), {}, Q88(data_width=data_width, fraction=fraction))

    def test_emitter_luts_match_module(self) -> None:
        e = self._emitter()
        assert e._exp_lut_entries() == tables.exp_lut_entries(16, 8)
        assert e._log_lut_entries() == tables.log_lut_entries(8)
        assert e._sqrt_lut_entries() == tables.sqrt_lut_entries(8)
        assert e._tanh_lut_entries() == tables.tanh_lut_entries(8)
        assert e._cosh_lut_entries() == tables.cosh_lut_entries(16, 8)
        assert e._exprel_lut_entries() == tables.exprel_lut_entries(16, 8)
        assert e._sigmoid_lut_entries() == tables.sigmoid_lut_entries(8)
        assert e._sin_lut_entries() == tables.sin_lut_entries(8)
        assert e._cos_lut_entries() == tables.cos_lut_entries(8)
        assert e._cbrt_lut_entries() == tables.cbrt_lut_entries(8)

    def test_emitter_sample_points_match_module(self) -> None:
        assert self._emitter()._sym_points() == tables.symmetric_sample_points()

    def test_emitter_const_float_delegates(self) -> None:
        node = ast.parse("1.0 / 3.0", mode="eval").body
        assert _VerilogExprEmitter._const_float(node) == tables.const_float(node)
