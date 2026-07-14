# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation-to-Verilog compiler facade

"""Stable equation-to-Verilog compiler import surface.

Registered modules and folded processing elements share one fixed-point
combinational core. Their implementations live in responsibility-specific
modules while this facade preserves the historical public import path.
"""

from __future__ import annotations

from ._verilog_folded_datapath import compile_to_datapath
from ._verilog_registered_module import compile_to_verilog

compile_to_datapath.__module__ = __name__
compile_to_verilog.__module__ = __name__

__all__ = ["compile_to_datapath", "compile_to_verilog"]
