# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pipeline stage analysis

"""Pipeline stage analysis utilities.

Computes critical path depth and required pipeline stages for
high-frequency targets.
"""

from __future__ import annotations

import ast
import math
from typing import Any


def _mul_div_depth(node: ast.AST) -> int:
    """Return the longest chain of Mult/Div operations from root to leaf.

    Parameters
    ----------
    node : ast.AST
        Root AST node.

    Returns
    -------
    int
        Maximum multiplicative depth.
    """
    if isinstance(node, ast.BinOp):
        left_d = _mul_div_depth(node.left)
        right_d = _mul_div_depth(node.right)
        is_mul = isinstance(node.op, (ast.Mult, ast.Div))
        return max(left_d, right_d) + (1 if is_mul else 0)
    if isinstance(node, ast.UnaryOp):
        return _mul_div_depth(node.operand)
    return 0


def critical_path_depth(expr_str: str) -> int:
    """Count the longest chain of Mult/Div nodes in an expression.

    This determines how many DSP blocks are chained in series in the
    resulting Verilog datapath. At high frequencies, each DSP in series
    adds ~2.5 ns of combinational delay.

    Parameters
    ----------
    expr_str : str
        Python-syntax arithmetic expression.

    Returns
    -------
    int
        Maximum multiply/divide chain length (0 = no multiplies).

    Examples
    --------
    >>> critical_path_depth("a * b + c")
    1
    >>> critical_path_depth("a * b * c * d")
    3
    >>> critical_path_depth("a + b")
    0
    """
    tree = ast.parse(expr_str, mode="eval")
    return _mul_div_depth(tree.body)


def pipeline_stages_needed(
    depth: int,
    target_freq_mhz: int,
    dsp_delay_ns: float = 2.5,
    routing_overhead_ns: float = 0.5,
) -> int:
    """Compute how many pipeline registers are needed between DSP blocks.

    Parameters
    ----------
    depth : int
        Critical path depth (from ``critical_path_depth()``).
    target_freq_mhz : int
        Target clock frequency in MHz.
    dsp_delay_ns : float
        Propagation delay per DSP multiply (default 2.5 ns for Artix-7).
    routing_overhead_ns : float
        Routing overhead per stage.

    Returns
    -------
    int
        Number of pipeline registers to insert (0 = no pipelining needed).
    """
    if depth == 0 or target_freq_mhz <= 0:
        return 0

    period_ns = 1000.0 / target_freq_mhz
    total_delay = depth * (dsp_delay_ns + routing_overhead_ns)

    if total_delay <= period_ns:
        return 0

    # Each pipeline register breaks the path into (stages+1) segments
    # Need ceil(total_delay / period) - 1 registers
    stages = math.ceil(total_delay / period_ns) - 1
    return max(0, stages)


def pipeline_analysis(
    equations: dict[str, str],
    target_freq_mhz: int = 100,
    dsp_delay_ns: float = 2.5,
) -> dict[str, dict[str, Any]]:
    """Analyse pipeline requirements for a multi-ODE system.

    Parameters
    ----------
    equations : dict
        Variable name → RHS expression.
    target_freq_mhz : int
        Target clock frequency.
    dsp_delay_ns : float
        DSP propagation delay.

    Returns
    -------
    dict
        Per-variable analysis: ``{var: {"depth": int, "stages": int,
        "achievable_mhz": int}}``.
    """
    results: dict[str, dict[str, Any]] = {}
    for var, expr in equations.items():
        depth = critical_path_depth(expr)
        stages = pipeline_stages_needed(depth, target_freq_mhz, dsp_delay_ns)
        if depth > 0:
            per_stage_delay = dsp_delay_ns + 0.5  # routing overhead
            achievable = int(1000.0 / (depth * per_stage_delay))
        else:
            achievable = target_freq_mhz
        results[var] = {
            "depth": depth,
            "stages": stages,
            "achievable_mhz": achievable,
        }
    return results
