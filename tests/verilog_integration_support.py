# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_verilog_integration.py

from __future__ import annotations

"""Test new VerilogGenerator emit methods and Halton IR resolution."""
from sc_neurocore.hdl_gen.verilog_generator import (
    VerilogGenerator,
    emit_sources_from_ir,
)

__all__ = ['VerilogGenerator', 'emit_sources_from_ir']
