# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog expression emitter test support

"""Shared imports and state variables for Verilog expression lowering tests."""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.compiler.verilog_expr_emitter import _emit_expr

_STATE_VARS = {"v": "v"}

__all__ = ["Q88", "_STATE_VARS", "_emit_expr", "pytest"]
