# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_e2e_pipeline.py

from __future__ import annotations

"""End-to-end integration tests — full compilation pipeline.

These tests exercise cross-cutting paths through the compiler that span
multiple modules (ODE → Verilog → constraints → drivers → formal → safety).

Run selectively with::

    pytest tests/e2e/ -m e2e -v
"""
import re
import pytest

LIF_EQUATIONS = {"v": "-(v - v_rest) / tau_m + R * I / C"}
IZH_EQUATIONS = {
    "v": "0.04 * v * v + 5 * v + 140 - u + I",
    "u": "a * (b * v - u)",
}
STATE_VARS_LIF = ["v"]
STATE_VARS_IZH = ["v", "u"]

__all__ = ["re", "pytest", "LIF_EQUATIONS", "IZH_EQUATIONS", "STATE_VARS_LIF", "STATE_VARS_IZH"]
