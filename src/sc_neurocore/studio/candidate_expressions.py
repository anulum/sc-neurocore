# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Candidate expression call admission

"""Check candidate calls against the actual equation namespace without running them.

The sandbox validates syntax and capabilities. This separate admission check
rejects calls whose argument counts cannot execute. NumPy ufuncs accept their
input arguments and optional positional outputs; those outputs remain supported.
The helpers retain their actual Python signatures. No expression is evaluated,
so checking a candidate cannot consume random samples or advance its state.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np

from sc_neurocore.neurons.equation_namespace import build_eval_namespace
from sc_neurocore.neurons.equation_refusals import EquationRefusal


def validate_candidate_calls(tree: ast.AST) -> None:
    """Refuse a direct call that the equation namespace cannot accept.

    ``tree`` must already have passed the equation safety gate. Unknown names
    are reported by candidate symbol validation before this check. Calls on
    attributes retain the sandbox's existing semantics.

    Parameters
    ----------
    tree:
        Candidate expression syntax tree approved by the equation safety gate.

    Returns
    -------
    None
        No expression is evaluated and no state or random stream is advanced.

    Raises
    ------
    EquationRefusal
        If a direct callee is not a namespace function or its argument count
        cannot be accepted by that function.
    """
    namespace = build_eval_namespace()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        function = namespace.get(name)
        if not callable(function):
            raise EquationRefusal(f"{name!r} is not an equation function")
        count = len(node.args)
        if isinstance(function, np.ufunc):
            accepted = function.nin <= count <= function.nargs
        elif function in (min, max):
            # Their one-iterable and multiple-scalar overloads both remain valid.
            accepted = count >= 1
        elif function is np.clip:
            # The AST gate excludes keywords. The positional form requires both
            # bounds (each may be None), followed by an optional output array.
            accepted = 3 <= count <= 4
        else:
            try:
                inspect.signature(function).bind(*([None] * count))
            except TypeError:
                accepted = False
            else:
                accepted = True
        if not accepted:
            raise EquationRefusal(f"function {name!r} does not accept {count} positional arguments")
