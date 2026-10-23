# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native equation neuron transactions

"""Wire model state, integration, events and logical clocks into one native graph."""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ...hdl_gen._ident import sanitize_ident
from ...neurons.equation_builder import EquationNeuron
from ..equation_lowering import equation_expressions, prepare_neuron
from .arithmetic import EquationArithmetic
from .clock import EquationClock, build_clock
from .events import lower_events
from .integrators import integrate
from .scheduling import EquationSchedule


@dataclass(frozen=True)
class EquationTransaction:
    """Describe the committed model words and cadence appended to a native graph.

    Attributes
    ----------
    states : dict of str to int
        Source state names bound to their current signed native registers.
    outputs : dict of str to int
        Public output names bound to actual state, spike and graded registers.
    clock : EquationClock
        Logical timestep and macro-boundary conditions for this model.
    pipeline_registers : int
        Number of actual selected wide product registers.
    """

    states: dict[str, int]
    outputs: dict[str, int]
    clock: EquationClock
    pipeline_registers: int


def lower_neuron(
    neuron: EquationNeuron,
    arithmetic: EquationArithmetic,
    commit: Callable[[int, int], int],
    *,
    event_commit: Callable[[int, int], int] | None = None,
    pipeline_stages: int = 0,
    pipeline_points: Sequence[str] = (),
) -> EquationTransaction:
    """Append a complete maintained neuron's registered timestep transaction.

    Parameters
    ----------
    neuron : EquationNeuron
        Authored model. Intrinsics are prepared without altering its live state.
    arithmetic : EquationArithmetic
        Destination builder and fixed-point numeric contract.
    commit : callable
        Apply the caller's declared overflow policy to a raw SSA word and its
        physical width. The result must have the configured signed Q word type.
        Runtime trap ownership for final integration belongs to this policy.
    event_commit : callable or None, optional
        Reset and graded-expression commit policy. None preserves the supplied
        commit callback for existing callers; the public model compiler supplies
        a separate policy to retain wrapping for events under overflow=trap.
    pipeline_stages : int, optional
        Zero leaves products combinational; a positive count enables the existing
        compiler's conservative one-register-per-selected-product policy.
    pipeline_points : sequence of str, optional
        Explicit maintained product names, mutually exclusive with global mode.

    Returns
    -------
    EquationTransaction
        Registers, output bindings and logical clocks. This component does not
        construct provenance or substitute for the public source compiler result.

    Raises
    ------
    TypeError
        Pipeline stages are not an integer or selection names are not strings.
    ValueError
        Pipeline selection, output names or the model's macro cadence is invalid.
    NotImplementedError
        Unsigned state, stochastic rounding, pipelined stochastic thresholds or
        macro modes unsupported by the maintained registered compiler are requested.
    """
    q, b = arithmetic.q, arithmetic.builder
    if not q.signed:
        raise NotImplementedError("unsigned equation state is not supported")
    if q.rounding not in {"truncate", "nearest", "bankers"}:
        raise NotImplementedError("equation products require truncate, nearest or bankers rounding")
    if type(pipeline_stages) is not int:
        raise TypeError("pipeline_stages must be an integer")
    if pipeline_stages < 0:
        raise ValueError("pipeline_stages must be non-negative")
    if isinstance(pipeline_points, (str, bytes)):
        raise TypeError("pipeline_points must be a sequence of names")
    if any(type(point) is not str for point in pipeline_points):
        raise TypeError("pipeline_points entries must be strings")
    if len(pipeline_points) != len(set(pipeline_points)):
        raise ValueError("pipeline_points must not contain duplicates")
    if pipeline_stages and pipeline_points:
        raise ValueError("global and explicit multiply pipelining are mutually exclusive")
    names = {name: sanitize_ident(name, context="state variable") for name in neuron.equations}
    if "spike" in names.values() or neuron.spike_value_expr and "spike_value" in names.values():
        raise ValueError("state output collides with a spike output")
    if neuron.substeps > 1 and not getattr(neuron, "_edge_detection", False):
        raise NotImplementedError("macro substeps require crossing detection without resets")
    neuron = prepare_neuron(neuron)
    if neuron.dt != 0.0:
        q.check_time_step(neuron.dt, neuron.method)
        if q.exact_code(neuron.dt) == 0:
            raise ValueError("equation dt underflows the configured Q format")
    width, ty = q.data_width, f"i{q.data_width}"
    states = {
        name: b.state(arithmetic.literal(neuron.initial_state[name], width), ty)
        for name in neuron.equations
    }
    bindings = {"I": b.input("I_t", ty)}
    bindings.update(
        {
            name: arithmetic.literal(value, width)
            for name, value in {**neuron.parameters, **neuron.constants}.items()
        }
    )
    externs: dict[str, int] = {}
    counter = None
    if neuron.uses_random_draws:
        counter = b.state(arithmetic.constant(0, 64), "i64")
        externs = {
            "step": counter,
            "seed": arithmetic.constant(neuron.random_seed, 64),
            "stream": arithmetic.constant(neuron.random_stream, 64),
        }
    schedule = EquationSchedule(
        q, b.word_delay, pipeline=pipeline_stages > 0, points=pipeline_points
    )
    candidates = integrate(neuron, arithmetic, states, bindings, commit, externs, schedule)
    if schedule.register_count and getattr(neuron, "_stochastic_threshold_enabled", False):
        raise NotImplementedError("stochastic thresholds do not support multiply pipelining")
    clock = build_clock(b, pipeline_registers=schedule.register_count, substeps=neuron.substeps)
    previous = states
    if neuron.substeps > 1:
        referenced = {
            node.id
            for _, expression in equation_expressions(neuron)
            for node in ast.walk(ast.parse(expression, mode="eval"))
            if isinstance(node, ast.Name)
        }
        previous = dict(states)
        for name in neuron.equations:
            if f"{name}_prev" in referenced:
                history = b.state(arithmetic.literal(neuron.initial_state[name], width), ty)
                b.store(history, candidates[name], clock.macro_boundary)
                previous[name] = history
    if counter is not None:
        b.store(
            counter,
            b.word_binary("add", counter, arithmetic.constant(1, 64), "i64"),
            clock.macro_boundary,
        )
    event = lower_events(
        neuron,
        arithmetic,
        candidates,
        previous,
        bindings,
        commit if event_commit is None else event_commit,
        clock.macro_boundary,
        externs,
    )
    outputs: dict[str, int] = {}
    for name, state in states.items():
        b.store(state, event.states[name], clock.advance)
        outputs[f"{names[name]}_out"] = state
    spike = b.state(b.constant_i64(0, "bool"), "bool")
    b.store(spike, event.spike)
    outputs["spike_out"] = spike
    if event.value is not None:
        graded = b.state(arithmetic.constant(0, width), ty)
        b.store(graded, event.value)
        outputs["spike_value_out"] = graded
    if schedule.register_count:
        outputs["latency"] = b.constant_i64(
            schedule.register_count, f"u{schedule.register_count.bit_length()}"
        )
    for name, value in outputs.items():
        b.output(name, value)
    return EquationTransaction(states, outputs, clock, schedule.register_count)
