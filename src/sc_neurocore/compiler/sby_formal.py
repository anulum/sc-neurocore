# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SymbiYosys formal verification flow

"""SymbiYosys formal verification utilities for compiled neuron modules.

Generates .sby configuration files for bounded model checking (BMC),
induction proofs, and reachability (cover) analysis.

For an end-to-end *machine-checked* equivalence proof — building the miter,
invoking ``sby``, and parsing the verdict — use
:func:`sc_neurocore.compiler.equivalence_check.prove_equivalence`, which drives
this flow to completion rather than only emitting the script.
"""

from __future__ import annotations

from typing import Literal


def generate_sby_script(
    module_name: str,
    *,
    sva_file: str | None = None,
    depth: int = 20,
    mode: Literal["bmc", "prove", "cover"] = "bmc",
    solver: str = "smtbmc",
    engine: str = "boolector",
) -> str:
    """Generate a SymbiYosys ``.sby`` formal verification script.

    Enables one-command bounded model checking of compiled neurons
    using open-source formal tools (SymbiYosys + Yosys + solver).

    Parameters
    ----------
    module_name : str
        Top-level Verilog module name.
    sva_file : str, optional
        SystemVerilog assertions file. Defaults to ``{module}_sva.sv``.
    depth : int
        BMC / induction depth in clock cycles.
    mode : str
        ``"bmc"`` (bounded), ``"prove"`` (induction), ``"cover"``.
    solver : str
        Solver backend (``"smtbmc"``, ``"aiger"``).
    engine : str
        SMT engine (``"boolector"``, ``"z3"``, ``"yices"``).

    Returns
    -------
    str
        Complete ``.sby`` configuration file.
    """
    if sva_file is None:
        sva_file = f"{module_name}_sva.sv"
    verilog_file = f"{module_name}.v"

    return (
        f"# Auto-generated SymbiYosys script for {module_name}\n"
        f"# SC-NeuroCore formal verification flow\n"
        f"# Run: sby {module_name}.sby\n"
        f"\n"
        f"[options]\n"
        f"mode {mode}\n"
        f"depth {depth}\n"
        f"\n"
        f"[engines]\n"
        f"{solver} {engine}\n"
        f"\n"
        f"[script]\n"
        f"read_verilog -formal {verilog_file}\n"
        f"read_verilog -sv -formal {sva_file}\n"
        f"prep -top {module_name}\n"
        f"\n"
        f"[files]\n"
        f"{verilog_file}\n"
        f"{sva_file}\n"
    )
