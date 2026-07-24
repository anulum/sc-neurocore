# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_c_expr_emitter.py

from __future__ import annotations

"""Tests for the C++ (ap_fixed) expression emitter."""
import pytest
from sc_neurocore.compiler.c_expr_emitter import CExprEmitter, emit_c_expr


def _emit(expr: str, state_vars: set[str] | None = None, **kw: object) -> str:
    code, _ = emit_c_expr(expr, state_vars or set(), **kw)  # type: ignore[arg-type]
    return code


__all__ = ["pytest", "CExprEmitter", "emit_c_expr", "_emit"]
