# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation-to-native-IR compiler

"""Compile maintained equation models into verified native scalar and state graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..hdl_gen._ident import sanitize_ident
from ..neurons.equation_builder import EquationNeuron
from .native_equation.arithmetic import EquationArithmetic
from .native_equation.commits import EquationCommits
from .native_equation.division import validate_division
from .native_equation.source import model_source
from .native_equation.transactions import lower_neuron
from .verilog_compiler_config import Q88

if TYPE_CHECKING:
    from sc_neurocore_engine.ir import ScGraph


@dataclass(frozen=True)
class EquationIR:
    """A verified native graph and the cadence of its registered model outputs.

    Attributes
    ----------
    graph : ScGraph
        Actual native graph. Its source payload binds the complete model and
        numerical choices; both graph text and source have native SHA256 checks.
    clocks_per_timestep : int
        Physical clocks between complete macro timesteps, including selected
        product-register fill or model substeps.
    pipeline_registers : int
        Actual selected wide product registers, rather than a requested count.
    output_widths : tuple of tuple of str and int
        Public output port names and physical widths in declaration order.
    """

    graph: ScGraph
    clocks_per_timestep: int
    pipeline_registers: int
    output_widths: tuple[tuple[str, int], ...]


def compile_to_ir(
    neuron: EquationNeuron,
    module_name: str = "sc_equation_neuron",
    data_width: int = 16,
    fraction: int = 8,
    *,
    signed: bool = True,
    overflow: str = "saturate",
    rounding: str = "truncate",
    pipeline_stages: int = 0,
    pipeline_points: list[str] | None = None,
    function_lowering: str = "lut-v1",
    random_precision: str = "standard",
) -> EquationIR:
    """Lower an equation model to typed native IR without embedding generated RTL.

    Parameters
    ----------
    neuron : EquationNeuron
        Authored equations, initial state, parameters, constants and events.
        Compilation preserves the caller's live state and RNG counters.
    module_name : str, optional
        Hardware module identifier, validated by the native graph verifier.
    data_width : int, optional
        Signed fixed-point state and input width, from 1 through 64. Internal
        expression words may be wider than the native scalar port limit.
    fraction : int, optional
        Number of fractional bits.
    signed : bool, optional
        Must be True, matching the maintained equation datapath.
    overflow : str, optional
        Saturate, wrap or simulation-only integrated-state trap. Reset and
        graded values retain wrapping under trap mode.
    rounding : str, optional
        Truncate, nearest or bankers product rounding.
    pipeline_stages : int, optional
        A positive count selects all maintained product registers.
    pipeline_points : list of str or None, optional
        Explicit product names, mutually exclusive with global pipelining.
    function_lowering : str, optional
        Existing lut-v1 default or opt-in structured v2 function programs.
    random_precision : str, optional
        Standard or exact normal-draw lowering.

    Returns
    -------
    EquationIR
        Verified registered graph, physical output widths and timestep cadence.
        Graph methods emit direct SystemVerilog and structured CIRCT HW/Comb/Seq.

    Raises
    ------
    ValueError
        The numerical, source, division, cadence or native graph contract fails.
    TypeError
        Geometry or pipeline arguments have incorrect runtime types.
    NotImplementedError
        The request uses an unsupported maintained datapath mode.
    ImportError
        The native engine providing scalar/state IR is unavailable.
    """
    from sc_neurocore_engine.ir import ScGraphBuilder

    q = Q88(data_width, fraction, signed, overflow, rounding, function_lowering, random_precision)
    if q.data_width > 64:
        raise NotImplementedError("native equation ports require data_width <= 64")
    validate_division(neuron, q)
    name = sanitize_ident(module_name, context="module name")
    builder = ScGraphBuilder(name)
    arithmetic = EquationArithmetic(builder, q)
    commits = EquationCommits(arithmetic)
    transaction = lower_neuron(
        neuron,
        arithmetic,
        commits.state,
        event_commit=commits.event,
        pipeline_stages=pipeline_stages,
        pipeline_points=() if pipeline_points is None else pipeline_points,
    )
    builder.bind_source(
        model_source(neuron, q, name, pipeline_stages, tuple(pipeline_points or ()))
    )
    graph = builder.build()
    if diagnostics := graph.verify():
        raise ValueError("invalid equation IR: " + "; ".join(diagnostics))
    outputs = tuple(
        (
            output,
            transaction.pipeline_registers.bit_length()
            if output == "latency"
            else 1
            if output == "spike_out"
            else data_width,
        )
        for output in transaction.outputs
    )
    return EquationIR(
        graph, transaction.clock.clocks_per_timestep, transaction.pipeline_registers, outputs
    )


def recompile_ir(text: str) -> EquationIR:
    """Replay a bound model and require its canonical native graph to match.

    Parameters
    ----------
    text : str
        Native graph text produced by :func:`compile_to_ir`. Native source and
        graph checksums are validated before reconstructing the model.

    Returns
    -------
    EquationIR
        Freshly compiled identical graph and its actual cadence/output contract.
        Equality establishes source consistency, without authenticating authorship.

    Raises
    ------
    ValueError
        Native text, source schema or replayed model does not match the graph.
    ImportError
        The native engine or a model's optional units dependency is unavailable.
    """
    from sc_neurocore_engine.ir import parse_ir

    from .native_equation.replay import restore_source

    original = parse_ir(text)
    payload = original.source_payload
    if payload is None:
        raise ValueError("native graph has no equation source contract")
    neuron, q, name, stages, points = restore_source(payload)
    result = compile_to_ir(
        neuron,
        name,
        q.data_width,
        q.fraction,
        signed=q.signed,
        overflow=q.overflow,
        rounding=q.rounding,
        pipeline_stages=stages,
        pipeline_points=points,
        function_lowering=q.function_lowering,
        random_precision=q.random_precision,
    )
    if result.graph.to_text() != original.to_text():
        raise ValueError("equation source replay does not match native graph")
    return result
