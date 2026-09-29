# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Deterministic dense IF replay and complete state traces

"""Replay explicit input frames using ordered float64 dense IF arithmetic."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .if_parameters import FloatArray, IFParameters
from .if_resources import DEFAULT_WORKING_BYTES
from .if_inputs import prepare_replay


@dataclass(frozen=True)
class IFReplayResult:
    """Owned output responses, final states and optional complete traces.

    A spiking final layer produces spike counts; a linear final layer produces
    its integrated current. Final states include the linear readout integrator.
    Trace tuples contain every layer's post-reset state and only IF spike events.
    """

    output: FloatArray
    final_state: tuple[FloatArray, ...]
    state_trace: tuple[FloatArray, ...]
    spike_trace: tuple[FloatArray, ...]
    numerical_profile: str = "dense-if-f64-sequential-v1"


def replay_dense_if(
    parameters: IFParameters,
    inputs: npt.ArrayLike,
    *,
    initial_state: Sequence[npt.ArrayLike] | None = None,
    trace: bool = False,
    binary_inputs: bool = True,
    max_working_bytes: int = DEFAULT_WORKING_BYTES,
) -> IFReplayResult:
    """Replay real input frames with inclusive thresholds and subtractive reset.

    Parameters
    ----------
    parameters : IFParameters
        Owned checked coefficients and final response mode.
    inputs : array_like
        Explicit ``(steps, batch, input_neurons)`` frames in ``[0, 1]``.
    initial_state : sequence of array_like, optional
        Finite ``(batch, output_neurons)`` states, one per layer. Caller-owned
        buffers are copied. Default IF states use the configured membrane shift;
        a linear readout starts from zero.
    trace : bool
        Retain each post-step state and every IF event when True.
    binary_inputs : bool
        Require exact zero/one input events. False admits bounded current drive.

    max_working_bytes : int
        Numeric buffer reservation checked before copying frames or states.

    Returns
    -------
    IFReplayResult
        Incremental spike counts or cumulative linear readout and owned states.

    Raises
    ------
    ValueError
        If input shape, values or initial-state dimensions are invalid.
    FloatingPointError
        If a finite-input replay overflows its numerical state.
    MemoryError
        If owned buffers exceed the declared working byte budget.

    Notes
    -----
    Input columns are accumulated in ascending order with separate float64
    multiply and add operations. Bias follows the complete dot product. Layers
    consume the preceding layer's events within the same timestep. Each IF
    emits at most one spike, including when the membrane equals its threshold.
    Empty time or batch axes preserve initial states and emit no events.
    A linear readout retains its supplied cumulative integral.
    """
    frames, states = prepare_replay(
        parameters, inputs, initial_state, trace, binary_inputs, max_working_bytes
    )
    steps, batch, _ = frames.shape
    layer_count = len(parameters.weights)
    spiking_layers = layer_count - int(parameters.output_mode == "linear")
    state_trace = (
        tuple(np.empty((steps, *state.shape), dtype=np.float64) for state in states)
        if trace
        else ()
    )
    spike_trace = (
        tuple(
            np.empty((steps, batch, parameters.weights[i].shape[0]), dtype=np.float64)
            for i in range(spiking_layers)
        )
        if trace
        else ()
    )
    output = np.zeros((batch, parameters.weights[-1].shape[0]), dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        for step in range(steps if batch else 0):
            layer_input = frames[step]
            for index, (weight, bias, threshold) in enumerate(
                zip(parameters.weights, parameters.biases, parameters.thresholds)
            ):
                current = np.zeros((batch, weight.shape[0]), dtype=np.float64)
                for column in range(weight.shape[1]):
                    product = layer_input[:, column, np.newaxis] * weight[np.newaxis, :, column]
                    current += product
                if bias is not None:
                    current += bias
                states[index] += current
                if index < spiking_layers:
                    events = (states[index] >= threshold).astype(np.float64)
                    states[index] -= events * threshold
                    layer_input = events
                    if trace:
                        spike_trace[index][step] = events
                    if index == layer_count - 1:
                        output += events
                if trace:
                    state_trace[index][step] = states[index]
    if parameters.output_mode == "linear":
        output = states[-1].copy()
    return IFReplayResult(output, tuple(states), state_trace, spike_trace)
