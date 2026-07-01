# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — lightweight dict-form NIR importer

"""Lightweight dictionary-form NIR importer.

This is a dependency-free convenience front-end for extracting ODE equations
from a plain ``{"nodes": {...}, "edges": [...]}`` description of a spiking graph
— handy for quick prototyping and stability sketches without the ``nir`` library
or a ``.nir`` file. Its per-node dynamics are derived from the **same** canonical
templates the FPGA back-end uses
(:data:`sc_neurocore.nir_bridge.neuron_templates.NEURON_TEMPLATES`), so the two
paths cannot disagree on what an LIF or CuBa-LIF neuron is.

For the authoritative import path — real typed ``nir.*`` graphs (or ``.nir``
files) parsed into an executable network and
:class:`~sc_neurocore.neurons.equation_builder.EquationNeuron` populations, with
affine/convolutional/pooling layers, subgraphs and hardware lowering — use
:func:`sc_neurocore.nir_bridge.from_nir`. This module intentionally covers only
the point-neuron node types and the plain-dict input form; it does not replace
the bridge.

Beyond the six NIR standard point-neuron types (``LIF``, ``IF``, ``LI``,
``CubaLIF``, ``CubaLI``, ``I``/integrator) this importer also recognises
``Izhikevich``, which is not part of the NIR node set and is provided here as a
documented extension.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ...nir_bridge.neuron_templates import NEURON_TEMPLATES

# Izhikevich is not a NIR standard node (the authoritative bridge does not map
# it); it is offered here as an explicit extension with its published dynamics.
_IZHIKEVICH_TEMPLATE: dict[str, Any] = {
    "equations": [
        "du/dt = a * (b * v - u)",
        "dv/dt = 0.04 * v * v + 5 * v + 140 - u + I",
    ],
    "threshold": "v > 30",
    "reset": "v = c; u = u + d",
    "default_params": {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
}

# String type tags that appear in dict-form graphs (framework- and case-variant)
# mapped onto the canonical template keys. Unknown tags fall back to ``li``.
_TYPE_ALIASES: dict[str, str] = {
    "lif": "lif",
    "leakyintegrateandfire": "lif",
    "if": "if",
    "integrateandfire": "if",
    "li": "li",
    "leakyintegrator": "li",
    "cubalif": "cuba_lif",
    "cubali": "cuba_li",
    "i": "integrator",
    "integrator": "integrator",
    "izhikevich": "izhikevich",
    "izh": "izhikevich",
}

_FALLBACK_TYPE = "li"


@dataclass(frozen=True)
class NIRGraph:
    """Imported dict-form NIR graph representation.

    Attributes
    ----------
    nodes : dict[str, dict]
        Node name → its raw input parameters.
    edges : list[tuple[str, str]]
        Directed edges ``(source, target)``.
    equations : dict[str, str]
        Node name → the membrane (``dv/dt``) right-hand side, with parameters
        substituted to concrete values. Kept as a flat ``str`` per node for
        back-compatibility and stability analysis.
    framework : str
        Source framework label.
    node_types : dict[str, str]
        Node name → the canonical template type it resolved to.
    state_equations : dict[str, dict[str, str]]
        Node name → ``{state_variable: right-hand side}`` for the full (possibly
        multi-compartment) model, so nothing is lost for multi-state neurons.
    thresholds : dict[str, str | None]
        Node name → its spike threshold expression (``None`` if the type has no
        threshold, e.g. leaky/plain integrators).
    resets : dict[str, str | None]
        Node name → its reset rule expression (``None`` if the type has none).
    parameters : dict[str, dict[str, float]]
        Node name → the resolved numeric parameters (template defaults overlaid
        with the node's own values).
    """

    nodes: dict[str, dict[str, Any]]
    edges: list[tuple[str, str]]
    equations: dict[str, str]
    framework: str
    node_types: dict[str, str] = field(default_factory=dict)
    state_equations: dict[str, dict[str, str]] = field(default_factory=dict)
    thresholds: dict[str, str | None] = field(default_factory=dict)
    resets: dict[str, str | None] = field(default_factory=dict)
    parameters: dict[str, dict[str, float]] = field(default_factory=dict)


def _canonical_type(raw: str) -> str:
    """Resolve a free-form node-type tag to a canonical template key."""
    key = re.sub(r"[^a-z0-9]", "", raw.lower())
    return _TYPE_ALIASES.get(key, _FALLBACK_TYPE)


def _template_for(ntype: str) -> dict[str, Any]:
    """Return the ODE template for a canonical type (Izhikevich is the extension)."""
    if ntype == "izhikevich":
        return _IZHIKEVICH_TEMPLATE
    return NEURON_TEMPLATES[ntype]


def _format_value(value: float) -> str:
    """Render a resolved parameter as a concrete numeric literal."""
    return repr(float(value))


def _substitute(expr: str, params: dict[str, float]) -> str:
    """Substitute parameter names with their numeric values (longest name first)."""
    for name in sorted(params, key=len, reverse=True):
        expr = re.sub(rf"\b{re.escape(name)}\b", _format_value(params[name]), expr)
    return expr


def _resolve_node(
    ntype: str, raw_params: dict[str, Any]
) -> tuple[dict[str, str], dict[str, float], str | None, str | None]:
    """Instantiate a template for one node: concrete state equations, params, threshold, reset."""
    template = _template_for(ntype)
    defaults: dict[str, float] = dict(template["default_params"])
    params = {
        **defaults,
        **{k: float(v) for k, v in raw_params.items() if k in defaults},
    }
    state_equations: dict[str, str] = {}
    for equation in template["equations"]:
        lhs, rhs = equation.split("=", 1)
        var = lhs.strip().removeprefix("d").split("/")[0]
        state_equations[var] = _substitute(rhs.strip(), params)
    threshold = _substitute(template["threshold"], params) if template["threshold"] else None
    reset = _substitute(template["reset"], params) if template["reset"] else None
    return state_equations, params, threshold, reset


def import_nir_graph(
    nir_data: dict[str, Any],
    *,
    framework: str = "snnTorch",
) -> NIRGraph:
    """Import a dict-form Neuromorphic Intermediate Representation graph.

    Each node's ``type`` selects a canonical ODE template (shared with the FPGA
    back-end); the node's parameters overlay the template defaults and are
    substituted into concrete equations, thresholds and reset rules. Node types
    outside the recognised set fall back to a leaky integrator.

    Parameters
    ----------
    nir_data : dict
        Graph as ``{"nodes": {name: {"type": ..., <params>}}, "edges": [...]}``.
        A node without a ``type`` defaults to ``LIF``.
    framework : str
        Source framework label recorded on the result.

    Returns
    -------
    NIRGraph
        Imported graph with per-node equations, state equations, thresholds,
        reset rules and resolved parameters. For the authoritative typed import
        use :func:`sc_neurocore.nir_bridge.from_nir`.
    """
    nodes = nir_data.get("nodes", {})
    edges = nir_data.get("edges", [])

    equations: dict[str, str] = {}
    node_types: dict[str, str] = {}
    state_equations: dict[str, dict[str, str]] = {}
    thresholds: dict[str, str | None] = {}
    resets: dict[str, str | None] = {}
    parameters: dict[str, dict[str, float]] = {}

    for name, params in nodes.items():
        ntype = _canonical_type(str(params.get("type", "LIF")))
        state_eqs, resolved, threshold, reset = _resolve_node(ntype, params)
        node_types[name] = ntype
        state_equations[name] = state_eqs
        equations[name] = state_eqs["v"]
        thresholds[name] = threshold
        resets[name] = reset
        parameters[name] = resolved

    return NIRGraph(
        nodes=nodes,
        edges=[(e[0], e[1]) for e in edges],
        equations=equations,
        framework=framework,
        node_types=node_types,
        state_equations=state_equations,
        thresholds=thresholds,
        resets=resets,
        parameters=parameters,
    )
