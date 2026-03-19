# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiler Package Init

from .equation_compiler import compile_to_verilog, equation_to_fpga, Q88

__all__ = ["compile_to_verilog", "equation_to_fpga", "Q88"]
