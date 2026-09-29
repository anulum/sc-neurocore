# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared dense IF input and initial-state admission

"""Own and validate explicit replay frames/states for every runtime backend."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from .if_parameters import FloatArray, IFParameters, real_array
from .if_resources import admit_replay_buffers


def prepare_replay(
    parameters: IFParameters,
    inputs: npt.ArrayLike,
    initial_state: Sequence[npt.ArrayLike] | None,
    trace: bool,
    binary_inputs: bool,
    max_working_bytes: int,
) -> tuple[FloatArray, list[FloatArray]]:
    """Admit and own complete frames and initial states before native execution.

    Parameters
    ----------
    parameters : IFParameters
        Checked owned finite dense coefficients and final response mode.
    inputs : array_like
        Explicit time/batch/input bounded currents or binary events.
    initial_state : sequence of array_like or None
        Finite batch/output state per layer; None selects admitted preloads.
    trace : bool
        Include complete trace storage in the numeric reservation.
    binary_inputs : bool
        Require exact zero/one events when True.
    max_working_bytes : int
        Positive addressable numeric-buffer limit excluding caller/runtime storage.

    Returns
    -------
    tuple
        Owned contiguous float64 frames and independently owned initial states.

    Raises
    ------
    ValueError
        Invalid frame/state dimensions, storage domains or input values.
    MemoryError
        Numeric reservation exceeds the working byte budget.
    FloatingPointError
        Finite preload multiplication overflows.
    """
    original = np.asarray(inputs)
    if original.ndim != 3 or original.shape[2] != parameters.weights[0].shape[1]:
        raise ValueError("input must have shape steps-by-batch-by-input-neurons")
    admit_replay_buffers(
        parameters.weights,
        parameters.biases,
        original.shape,
        trace=trace,
        linear=parameters.output_mode == "linear",
        max_working_bytes=max_working_bytes,
    )
    frames = real_array(original, "input")
    if bool(((frames < 0) | (frames > 1)).any()):
        raise ValueError("input values must be between zero and one")
    if binary_inputs and bool(((frames != 0) & (frames != 1)).any()):
        raise ValueError("spike input values must be zero or one")
    _, batch, _ = frames.shape
    layer_count = len(parameters.weights)
    if initial_state is not None and len(initial_state) != layer_count:
        raise ValueError("initial state must have one array per layer")
    states: list[FloatArray] = []
    spiking_layers = layer_count - int(parameters.output_mode == "linear")
    for index, (weight, threshold) in enumerate(zip(parameters.weights, parameters.thresholds)):
        shape = (batch, weight.shape[0])
        if initial_state is None:
            shift = (
                parameters.layer_membrane_fractions[index] * threshold
                if index < spiking_layers
                else 0.0
            )
            if not np.isfinite(shift):
                raise FloatingPointError("initial membrane overflow")
            state = np.full(shape, shift, dtype=np.float64)
        else:
            original_state = np.asarray(initial_state[index])
            if original_state.shape != shape:
                raise ValueError("initial state dimensions must match each layer and batch")
            state = real_array(original_state, "initial state")
        states.append(state)
    return frames, states
