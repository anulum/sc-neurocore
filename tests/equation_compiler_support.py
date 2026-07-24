# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_equation_compiler.py

from __future__ import annotations

"""Tests for equation_compiler: ODE strings → synthesizable Verilog RTL."""
import pint
from sc_neurocore.neurons.equation_builder import EquationNeuron, from_equations
from sc_neurocore.compiler.equation_compiler import (
    compile_to_verilog,
    equation_to_fpga,
)

UNIT_REGISTRY = pint.UnitRegistry()

__all__ = [
    "pint",
    "EquationNeuron",
    "from_equations",
    "compile_to_verilog",
    "equation_to_fpga",
    "UNIT_REGISTRY",
]
