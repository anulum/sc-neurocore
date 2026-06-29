# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation compiler facade

"""Compile arbitrary ODE neuron equations to synthesizable Verilog."""

from __future__ import annotations

from .fpga_wrapper import (
    equation_to_fpga,
)
from .testbench_gen import (
    generate_testbench,
)
from .verilog_compiler import (
    compile_to_datapath,
    compile_to_verilog,
)
from .verilog_compiler_config import (
    Q88,
)
from .verilog_expr_emitter import (
    _VerilogExprEmitter,
)

__all__ = [
    "Q88",
    "_VerilogExprEmitter",
    "compile_to_datapath",
    "compile_to_verilog",
    "equation_to_fpga",
    "generate_testbench",
]
