# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR/ONNX → FPGA network compiler
"""Neuron parameter binding and per-type RTL emission for FPGA compilation."""

from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from ..compiler.equation_compiler import compile_to_verilog
from ..hdl_gen._ident import sanitize_ident
from ..neurons.equation_builder import EquationNeuron, from_equations
from .neuron_graph import NeuronSpec
from .neuron_templates import NEURON_TEMPLATES

_NEURON_TEMPLATES = NEURON_TEMPLATES


def _representative_param(values: np.ndarray[Any, Any], label: str) -> float:
    """Return the reference (first-neuron) value of a per-neuron parameter.

    This value becomes the default of the shared, parameterised RTL neuron module.
    A heterogeneous population is no longer rejected: every neuron whose own
    quantised parameter differs from this default is instantiated with an explicit
    Verilog parameter override at the top level (see :func:`_neuron_param_override`).
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Empty parameter array for {label}")
    return float(arr[0])


def _type_default_qparams(pops: Sequence[NeuronSpec], data_width: int) -> dict[str, dict[str, int]]:
    """First-population, first-neuron quantised parameters per neuron type.

    Values are stored as the unsigned two's-complement bit pattern the neuron
    module declares as its parameter default, so the top level only overrides a
    neuron whose quantised parameter differs.
    """
    mask = (1 << data_width) - 1
    defaults: dict[str, dict[str, int]] = {}
    for pop in pops:
        if pop.neuron_type in defaults:
            continue
        entry: dict[str, int] = {}
        for pname, pval in pop.params.items():
            arr = np.atleast_1d(np.asarray(pval).reshape(-1))
            entry[pname] = int(arr[0]) & mask
        defaults[pop.neuron_type] = entry
    return defaults


def _neuron_param_override(
    pop: NeuronSpec,
    neuron_idx: int,
    type_defaults: dict[str, dict[str, int]],
    data_width: int,
) -> str:
    """Return a Verilog parameter-override clause for a single neuron instance.

    Empty when this neuron's per-neuron quantised parameters all equal the shared
    module defaults (the homogeneous case, so the emitted RTL is unchanged).
    Otherwise emits ``#(.P_X(W'sdN), ...)`` so a heterogeneous population reuses
    the same parameterised module with each neuron's own quantised parameters. The
    literal is the unsigned two's-complement bit pattern, matching how the module
    declares each parameter default (negative fixed-point values included).
    """
    mask = (1 << data_width) - 1
    defaults = type_defaults.get(pop.neuron_type, {})
    fragments: list[str] = []
    for pname in sorted(pop.params):
        if pname not in defaults:
            continue
        arr = np.atleast_1d(np.asarray(pop.params[pname]).reshape(-1))
        raw = int(arr[neuron_idx]) if arr.shape[0] == pop.n_neurons else int(arr[0])
        qval = raw & mask
        if qval != defaults[pname]:
            vname = f"P_{sanitize_ident(pname, context='parameter name').upper()}"
            fragments.append(f".{vname}({data_width}'sd{qval})")
    if not fragments:
        return ""
    return " #(" + ", ".join(fragments) + ")"


def _heterogeneous_param_names(pop: NeuronSpec, data_width: int) -> list[str]:
    """Return the sorted parameter names whose per-neuron quantised values vary in ``pop``.

    A name is heterogeneous when its per-neuron array (length ``n_neurons``) holds more
    than one distinct value at the quantised data width — exactly the set the direct path
    emits per-neuron ``#(.P_X(...))`` overrides for (see :func:`_neuron_param_override`),
    and the set the folded interconnect must stream through a per-neuron parameter ROM.

    Uniformity is decided at the *quantised* data width (the same mask the override
    detection uses), so two float parameters that round to the same fixed-point literal
    are not heterogeneous. A scalar / population-shared parameter (an array that is not
    per-neuron) is never heterogeneous.
    """
    mask = (1 << data_width) - 1
    names: list[str] = []
    for pname, pval in pop.params.items():
        arr = np.atleast_1d(np.asarray(pval).reshape(-1))
        if arr.shape[0] != pop.n_neurons:
            continue  # scalar / shared parameter — identical for every neuron
        if len({int(v) & mask for v in arr}) > 1:
            names.append(pname)
    return sorted(names)


def _population_params_are_uniform(pop: NeuronSpec, data_width: int) -> bool:
    """Return whether every per-neuron quantised parameter is uniform."""
    return not _heterogeneous_param_names(pop, data_width)


def _param_neuron_literal(pop: NeuronSpec, pname: str, neuron_idx: int, data_width: int) -> str:
    """Return the Verilog signed literal of ``pop``'s ``pname`` for one neuron (quantised).

    The literal is the unsigned two's-complement bit pattern the module declares its
    parameter default with — the same form the direct path's per-neuron overrides use
    (see :func:`_neuron_param_override`) — so the folded parameter ROM feeds the PE the
    identical value the direct instance would receive.
    """
    mask = (1 << data_width) - 1
    arr = np.atleast_1d(np.asarray(pop.params[pname]).reshape(-1))
    raw = int(arr[neuron_idx]) if arr.shape[0] == pop.n_neurons else int(arr[0])
    return f"{data_width}'sd{raw & mask}"


def _dequantised_pop(pop: NeuronSpec, fraction: int) -> NeuronSpec:
    """Return ``pop`` with its quantised parameter values scaled back to real units.

    A :class:`QuantisedGraph` population stores fixed-point *integer* parameters
    (``value × 2**fraction``). The folded PE and the per-instance module both encode
    real-valued parameters with :meth:`Q88.encode`, so they must be handed the *real*
    value — feeding the already-quantised integer encodes it a second time (a 16-bit
    ``tau = 5120`` re-encodes to ``5120 × 256 mod 2**16 = 0``, silently baking a broken
    parameter into the shared PE). The rescale is lossless for genuine fixed-point values
    (``5120 / 256 = 20.0`` re-encodes to ``5120``). Parameters absent from ``pop.params``
    are untouched (they fall back to the template default, already a real value).
    """
    if not pop.params:
        return pop
    scale = float(1 << fraction)
    rescaled = {
        name: np.asarray(values, dtype=np.float64) / scale for name, values in pop.params.items()
    }
    return replace(pop, params=rescaled)


def _resolved_population_params(neuron_type: str, pop: NeuronSpec) -> dict[str, float]:
    """Resolve population parameters without averaging per-neuron values."""
    template = _NEURON_TEMPLATES.get(neuron_type)
    if template is None:
        raise ValueError(f"No ODE template for neuron type: {neuron_type!r}")

    params: dict[str, float] = {
        name: float(value) for name, value in template["default_params"].items()
    }
    for pname, pval in pop.params.items():
        if pname not in params:
            raise ValueError(
                f"Parameter {pop.name}.{pname} is not supported by the "
                f"{neuron_type!r} FPGA template"
            )
        params[pname] = _representative_param(pval, f"{pop.name}.{pname}")
    return params


def _population_module_signature(pop: NeuronSpec) -> tuple[Any, ...]:
    """Build the exact parameter signature for shared module reuse."""
    params = _resolved_population_params(pop.neuron_type, pop)
    return (
        pop.neuron_type,
        float(pop.dt),
        tuple((name, params[name]) for name in sorted(params)),
    )


def build_neuron_module(
    neuron_type: str,
    pop: NeuronSpec,
    *,
    data_width: int = 16,
    fraction: int = 8,
) -> str:
    """Build a Verilog module for one canonical neuron type.

    Uses the existing ``equation_compiler.compile_to_verilog()`` with
    canonical ODE templates.

    Parameters
    ----------
    neuron_type : str
        Canonical neuron type (``"lif"``, ``"if"``, etc.).
    pop : NeuronSpec
        Representative population (for parameter defaults).
    data_width : int
        Fixed-point data width.
    fraction : int
        Fractional bits.

    Returns
    -------
    str
        Synthesisable Verilog module source.

    Raises
    ------
    ValueError
        If the neuron type or one of its parameters has no canonical FPGA
        template contract.
    """
    neuron = _population_neuron(neuron_type, pop)
    return compile_to_verilog(
        neuron,
        module_name=f"sc_nir_{neuron_type}",
        data_width=data_width,
        fraction=fraction,
    )


def _population_neuron(neuron_type: str, pop: NeuronSpec) -> EquationNeuron:
    """Build the canonical :class:`EquationNeuron` for one population's type.

    Single source of truth for the ODE/threshold/reset/params/init/dt used by
    both the per-instance module (:func:`build_neuron_module`) and the folded
    datapath PE (:func:`~fpga_folded_interconnect.build_folded_interconnect`),
    so the two share identical dynamics.
    """
    template = _NEURON_TEMPLATES[neuron_type]

    # Shared modules are only emitted when every per-neuron parameter is
    # identical.  Heterogeneous populations need generated per-neuron modules
    # and are rejected rather than averaged.
    params = _resolved_population_params(neuron_type, pop)

    init: dict[str, float] = {}
    for eq_str in template["equations"]:
        var_name = eq_str.split("/")[0].replace("d", "", 1).strip()
        if var_name == "v":
            init["v"] = params.get("v_leak", 0.0)
        else:
            init[var_name] = 0.0

    return from_equations(
        *template["equations"],
        threshold=template["threshold"],
        reset=template["reset"],
        params=params,
        init=init,
        dt=pop.dt,
    )
