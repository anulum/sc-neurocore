# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded deterministic IF input encoding

"""Encode finite input batches in bounded blocks and preserve complete IF state."""

from typing import Literal

import numpy as np
import numpy.typing as npt

from .if_dispatch import ReplayBackend, select_native
from .if_native import replay_native
from .if_parameters import FloatArray, IFParameters, real_array
from .if_replay import replay_dense_if
from .if_resources import admit_replay_buffers


def simulate_encoded(
    parameters: IFParameters,
    x: npt.ArrayLike,
    steps: int,
    input_mode: Literal["poisson", "constant"],
    seed: int,
    max_working_bytes: int,
    backend: ReplayBackend,
) -> FloatArray:
    """Run row-major MT19937 events or constant current with bounded storage.

    Parameters
    ----------
    parameters : IFParameters
        Owned and validated dense coefficients.
    x : array_like
        Finite input vector or batch in the unit interval.
    steps : int
        Positive timestep budget, at most 2**53.
    input_mode : {'poisson', 'constant'}
        Bernoulli events or direct bounded currents.
    seed : int
        Unsigned 32-bit MT19937 seed.
    max_working_bytes : int
        Positive numeric buffer budget, checked before encoding allocation.

    backend : {"auto", "numpy", "rust", "go", "mojo", "julia"}
        Replay runtime selected once for the complete encoded call, including empty batches.

    Returns
    -------
    ndarray
        Accumulated output with the original vector/batch shape.

    Raises
    ------
    ValueError
        If encoding, shape, domains, seed or timestep budget are invalid.
    MemoryError
        If a block's numeric buffers exceed the declared byte budget.
    """
    if type(steps) is not int or not 0 < steps <= 2**53:
        raise ValueError("T must be a positive integer no greater than 2**53")
    if input_mode not in ("poisson", "constant"):
        raise ValueError("input mode must be poisson or constant")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    native = select_native(backend)
    original = np.asarray(x)
    if original.ndim not in (1, 2) or original.shape[-1] != parameters.weights[0].shape[1]:
        raise ValueError("input dimensions must match the first layer")
    batch = 1 if original.ndim == 1 else original.shape[0]
    admit_replay_buffers(
        parameters.weights,
        parameters.biases,
        (min(64, steps), batch, original.shape[-1]),
        trace=False,
        linear=parameters.output_mode == "linear",
        max_working_bytes=max_working_bytes,
    )
    block_budget = max_working_bytes
    if native is not None and steps > 64:
        retained = (
            8
            * batch
            * (sum(w.shape[0] for w in parameters.weights) + parameters.weights[-1].shape[0])
        )
        block_budget -= retained
        admit_replay_buffers(
            parameters.weights,
            parameters.biases,
            (64, batch, original.shape[-1]),
            trace=False,
            linear=parameters.output_mode == "linear",
            max_working_bytes=block_budget,
        )
    values = real_array(original, "input")
    squeeze = values.ndim == 1
    if squeeze:
        values = values[np.newaxis]
    if bool(((values < 0) | (values > 1)).any()):
        raise ValueError("input values must be between zero and one")
    rng = np.random.RandomState(seed)
    state: tuple[FloatArray, ...] | None = None
    output = np.zeros((values.shape[0], parameters.weights[-1].shape[0]), dtype=np.float64)
    for start in range(0, steps if batch else 0, 64):
        shape = (min(64, steps - start), *values.shape)
        frames = (
            (rng.random(shape) < values).astype(np.float64)
            if input_mode == "poisson"
            else np.broadcast_to(values, shape)
        )
        if native is None:
            result = replay_dense_if(
                parameters,
                frames,
                initial_state=state,
                trace=False,
                binary_inputs=input_mode == "poisson",
                max_working_bytes=block_budget,
            )
        else:
            result = replay_native(
                native, parameters, frames, state, False, input_mode == "poisson", block_budget
            )
        state = result.final_state
        if parameters.output_mode == "linear":
            output = result.output
        else:
            output += result.output
    return output[0] if squeeze else output
