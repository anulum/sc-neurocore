# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Guard-bit auto-computation

"""Guard-bit auto-computation utilities.

Determines how many extra MSBs are needed in intermediate accumulators
to prevent silent overflow.
"""

from __future__ import annotations

import ast
import math


def _count_additions(expr_str: str) -> int:
    """Count the number of addition/subtraction nodes in an expression AST.

    Parameters
    ----------
    expr_str : str
        A Python-syntax arithmetic expression.

    Returns
    -------
    int
        Number of Add/Sub operations in the AST.
    """
    tree = ast.parse(expr_str, mode="eval")
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            count += 1
    return count


def compute_guard_bits(expr_str: str) -> int:
    """Compute the number of guard bits needed for safe accumulation.

    When summing N values, the accumulator needs ``ceil(log2(N+1))``
    extra MSBs to guarantee no intermediate overflow. For a single
    addition (a + b), 1 guard bit suffices. For ``a + b + c + d``,
    2 guard bits are needed.

    Parameters
    ----------
    expr_str : str
        A Python-syntax ODE right-hand-side expression.

    Returns
    -------
    int
        Minimum number of guard bits needed (0 if no additions).

    Examples
    --------
    >>> compute_guard_bits("a + b")
    1
    >>> compute_guard_bits("a + b + c + d")
    2
    >>> compute_guard_bits("a * b")
    0
    """
    n_adds = _count_additions(expr_str)
    if n_adds == 0:
        return 0
    # N additions can produce up to N+1 terms; guard = ceil(log2(N+1))
    return math.ceil(math.log2(n_adds + 1))


def compute_guard_bits_multi(equations: dict[str, str]) -> dict[str, int]:
    """Compute guard bits for every state variable in a multi-ODE system.

    Parameters
    ----------
    equations : dict
        Mapping from variable name to RHS expression string.

    Returns
    -------
    dict
        Mapping from variable name to required guard bits.
    """
    return {var: compute_guard_bits(expr) for var, expr in equations.items()}
