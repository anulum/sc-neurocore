# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — the diffusion-noise symbol xi lowered to the counter-based normal draw

"""Lower the diffusion-noise symbol ``xi`` to ``randn(XI_SITE) * (1/sqrt(dt))``.

In the Python model (:class:`~sc_neurocore.neurons.equation_builder.EquationNeuron`
without a ``noise_rng``) ``xi`` is the counter-based standard-normal draw of the
reserved call site :data:`~sc_neurocore.neurons.equation_random.XI_SITE`, multiplied
by ``1/sqrt(dt)``; every occurrence in a step reads the same draw. The compiler
reads exactly that: :func:`lower_diffusion_noise` returns a copy of the neuron
whose expressions spell every ``xi`` as ``(randn(XI_SITE) * c)`` with the model's
own float ``c``. Every draw of one site at one step is the same number, so the
copies agree with the shared sample, and the RTL, C and Rust emitters lower it
like any other ``randn()`` (the standard or the exact unit) and constant product.

Two cases are refused rather than lowered: ``xi`` drawn from a caller's
``noise_rng`` (a NumPy generator the hardware cannot reproduce) and ``xi`` with
``substeps > 1`` (each sub-step reads its own site, which the sub-stepped
datapath does not number).
"""

from __future__ import annotations

import ast
import copy
from typing import TypeVar

from sc_neurocore.neurons.equation_random import XI_SITE

__all__ = ["lower_diffusion_noise", "xi_text"]

_N = TypeVar("_N")


class _ReplaceXi(ast.NodeTransformer):
    """Replace every ``xi`` name by the draw expression."""

    def __init__(self, replacement: ast.expr) -> None:
        self._replacement = replacement

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Return the draw expression for ``xi`` and the name itself otherwise."""
        if node.id == "xi":
            return copy.deepcopy(self._replacement)
        return node


def xi_text(scale: float) -> str:
    """Return the expression ``xi`` stands for: ``(randn(XI_SITE) * scale)``.

    Parameters
    ----------
    scale : float
        ``1/sqrt(dt)`` as the model computes it.

    Returns
    -------
    str
        The draw expression, with ``scale`` written to round-trip exactly.
    """
    return f"(randn({XI_SITE}) * {scale!r})"


def _rewrite(expression: str | None, replacement: ast.expr) -> str | None:
    """Return ``expression`` with every ``xi`` replaced, or unchanged when it has none."""
    if expression is None:
        return None
    tree = ast.parse(expression, mode="eval")
    if not any(isinstance(node, ast.Name) and node.id == "xi" for node in ast.walk(tree)):
        return expression
    return ast.unparse(ast.fix_missing_locations(_ReplaceXi(replacement).visit(tree)))


def lower_diffusion_noise(neuron: _N) -> _N:
    """Return the neuron with ``xi`` spelled as its counter-based draw, for compilation.

    Parameters
    ----------
    neuron : EquationNeuron
        The neuron to compile. Returned unchanged when no expression names ``xi``.

    Returns
    -------
    EquationNeuron
        A shallow copy whose equations, reset rules, threshold, rate,
        probability, spike value and Jacobian expressions read
        ``(randn(XI_SITE) * (1/sqrt(dt)))`` for ``xi``.

    Raises
    ------
    ValueError
        If ``xi`` comes from a caller's ``noise_rng`` or the neuron sub-steps.
    """
    if not getattr(neuron, "uses_diffusion_noise", False):
        return neuron
    if getattr(neuron, "xi_from_noise_rng", False):
        raise ValueError(
            "xi is drawn from the caller's noise_rng, which has no RTL or kernel lowering; "
            "build the neuron without noise_rng to draw xi from the counter-based generator"
        )
    if int(getattr(neuron, "substeps", 1)) > 1:
        raise ValueError(
            "xi with substeps > 1 reads one call site per sub-step, which the sub-stepped "
            "datapath does not number; use substeps=1"
        )
    scale = float(vars(neuron)["_xi_scale"])
    replacement = ast.parse(xi_text(scale), mode="eval").body
    lowered = copy.copy(neuron)
    attributes = vars(lowered)
    attributes["equations"] = {
        name: str(_rewrite(text, replacement)) for name, text in attributes["equations"].items()
    }
    attributes["reset_rules"] = {
        name: str(_rewrite(text, replacement)) for name, text in attributes["reset_rules"].items()
    }
    attributes["jacobian_expressions"] = {
        name: str(_rewrite(text, replacement))
        for name, text in attributes.get("jacobian_expressions", {}).items()
    }
    for name in ("threshold_expr", "rate_expression", "probability_expression"):
        attributes[name] = _rewrite(attributes.get(name), replacement)
    attributes["spike_value_expr"] = _rewrite(attributes.get("spike_value_expr"), replacement)
    return lowered
