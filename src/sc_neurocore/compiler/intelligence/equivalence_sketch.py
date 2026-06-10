# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Formal equivalence sketch

"""Formal equivalence proof skeleton between ODE and RTL."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EquivalenceSketch:
    """Formal equivalence proof skeleton between ODE and RTL.

    Attributes
    ----------
    module_name : str
        Module under verification.
    equations : dict[str, str]
        Source ODE equations.
    assertions : list[str]
        SVA assertion strings for equivalence checking.
    proof_steps : list[str]
        Human-readable proof argument steps.
    quantisation_bound : float
        Maximum quantisation error bound.
    """

    module_name: str
    equations: dict[str, str]
    assertions: list[str]
    proof_steps: list[str]
    quantisation_bound: float


def generate_equivalence_sketch(
    module_name: str,
    equations: dict[str, str],
    *,
    data_width: int = 16,
    fraction: int = 8,
) -> EquivalenceSketch:
    """Generate a formal equivalence proof sketch for ODE→RTL translation.

    Produces a structured argument that the compiled Verilog computes
    the same function as the source ODE within quantisation error.

    Parameters
    ----------
    module_name : str
        Module name.
    equations : dict[str, str]
        ODE equations.
    data_width : int
        Fixed-point total width.
    fraction : int
        Fractional bits.

    Returns
    -------
    EquivalenceSketch
        Proof skeleton with SVA assertions.
    """
    lsb = 2.0 ** (-fraction)
    max_val = 2.0 ** (data_width - fraction - 1) - lsb
    q_bound = lsb / 2  # half-LSB quantisation error

    proof_steps = [
        f"1. Source ODE: {len(equations)} state variable(s)",
    ]
    for sv, expr in equations.items():
        proof_steps.append(f"   {sv}' = {expr}")

    proof_steps.extend(
        [
            f"2. Fixed-point format: Q{data_width - fraction - 1}.{fraction} "
            f"({data_width}-bit, LSB = {lsb})",
            f"3. Quantisation error bound: ε ≤ {q_bound} per operation",
            f"4. Range: [{-max_val - lsb}, {max_val}]",
            "5. Each arithmetic operation introduces ≤ ε truncation error",
            "6. For N chained operations, total error ≤ N × ε",
        ]
    )

    # Count operations for error accumulation
    total_ops = 0
    for expr in equations.values():
        total_ops += expr.count("+") + expr.count("-")
        total_ops += expr.count("*") + expr.count("/")

    accumulated_bound = total_ops * q_bound
    proof_steps.append(
        f"7. Total operations: {total_ops}, accumulated bound: {accumulated_bound:.2e}"
    )
    proof_steps.append(
        "8. CONCLUSION: RTL output matches ODE within accumulated "
        f"quantisation bound ε_total = {accumulated_bound:.2e}"
    )

    # SVA assertions
    assertions = []
    for sv in equations:
        assertions.append(
            f"assert property (@(posedge clk) disable iff (rst) "
            f"|{sv}_next - {sv}_ref| <= {int(accumulated_bound * (1 << fraction))});"
        )

    return EquivalenceSketch(
        module_name=module_name,
        equations=equations,
        assertions=assertions,
        proof_steps=proof_steps,
        quantisation_bound=accumulated_bound,
    )
