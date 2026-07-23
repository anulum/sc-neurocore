# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEmitterDelegationParity from former test_expr_lut_tables.py

"""Focused suite: TestEmitterDelegationParity from former test_expr_lut_tables.py."""

from __future__ import annotations

from tests.expr_lut_tables_support import *  # noqa: F403

class TestEmitterDelegationParity:
    """The Verilog emitter must return byte-identical tables after delegation."""

    def _emitter(self, data_width: int = 16, fraction: int = 8) -> _VerilogExprEmitter:
        return _VerilogExprEmitter({}, {}, Q88(data_width=data_width, fraction=fraction))

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
