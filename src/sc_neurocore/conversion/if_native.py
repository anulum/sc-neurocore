# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native owned dense IF replay adapter

"""Call the native ownership ABI without copying returned trajectories."""

import ctypes
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from .if_inputs import prepare_replay
from .if_native_types import (
    BufferCall,
    BufferView,
    FreeCall,
    LayerSpec,
    NativeAPI,
    NativeOwner,
    ReplayCall,
    ReplayRequest,
)
from .if_parameters import FloatArray, IFParameters
from .if_replay import IFReplayResult


@lru_cache(maxsize=16)
def load_native(path: str) -> NativeAPI:
    """Load a configured C library and require the exact replay ownership ABI.

    Parameters
    ----------
    path : str
        Explicit owner-configured native library; no build or download occurs.

    Returns
    -------
    NativeAPI
        Loaded ABI-one replay, buffer and release functions.

    Raises
    ------
    RuntimeError
        Library unavailable, entry points absent or ABI version incompatible.
    """
    try:
        library = ctypes.CDLL(str(Path(path).expanduser().resolve()))
        library.sc_if_abi_version.argtypes = []
        library.sc_if_abi_version.restype = ctypes.c_uint32
        if library.sc_if_abi_version() != 1:
            raise RuntimeError("native dense IF library has incompatible ownership ABI")
        library.sc_if_replay.argtypes = [
            ctypes.POINTER(ReplayRequest),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.sc_if_replay.restype = ctypes.c_int32
        library.sc_if_buffer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_size_t,
            ctypes.POINTER(BufferView),
        ]
        library.sc_if_buffer.restype = ctypes.c_int32
        library.sc_if_free.argtypes = [ctypes.c_void_p]
        library.sc_if_free.restype = None
    except (OSError, AttributeError) as error:
        raise RuntimeError(
            "native dense IF library or ownership entry points unavailable"
        ) from error
    return NativeAPI(
        library,
        cast(ReplayCall, library.sc_if_replay),
        cast(BufferCall, library.sc_if_buffer),
        cast(FreeCall, library.sc_if_free),
    )


def _view(owner: NativeOwner, kind: int, index: int, shape: tuple[int, ...]) -> FloatArray:
    """Bind one native result vector to an array retaining its supplying native owner."""
    view = BufferView()
    if owner.api.buffer(owner.handle, kind, index, ctypes.byref(view)) != 0:
        raise RuntimeError("native dense IF result buffer unavailable")
    if view.length != np.prod(shape, dtype=object) or (view.length and not view.data):
        raise RuntimeError("native dense IF result dimensions are invalid")
    array_type = ctypes.c_double * view.length
    storage = array_type.from_address(view.data) if view.length else array_type()
    storage.__dict__["_native_owner"] = owner
    values: FloatArray = np.ctypeslib.as_array(storage).reshape(shape)
    return values


def replay_native(
    api: NativeAPI,
    parameters: IFParameters,
    inputs: npt.ArrayLike,
    initial_state: Sequence[npt.ArrayLike] | None,
    trace: bool,
    binary_inputs: bool,
    max_working_bytes: int,
) -> IFReplayResult:
    """Replay through the checked native C boundary with automatically owned views.

    Parameters
    ----------
    api : NativeAPI
        Configured ABI-one native library.
    parameters : IFParameters
        Owned validated coefficients; borrowed throughout the native call.
    inputs : array_like
        Explicit time/batch/input unit currents or binary events.
    initial_state : sequence of array_like or None
        Optional independently copied batch/output states.
    trace : bool
        Retain complete state/event trajectories.
    binary_inputs : bool
        Require exact zero/one events when True.
    max_working_bytes : int
        Positive numeric reservation excluding caller/runtime overhead.

    Returns
    -------
    IFReplayResult
        Native-owned arrays whose views retain their supplying library/owner.

    Raises
    ------
    ValueError
        Invalid input, geometry, coefficients or native domain refusal.
    MemoryError
        Native or shared numeric buffer reservation refused.
    FloatingPointError
        Finite arithmetic overflow refused by the native kernel.
    RuntimeError
        Malformed native ownership or internal native failure.
    """
    frames, states = prepare_replay(
        parameters, inputs, initial_state, trace, binary_inputs, max_working_bytes
    )
    steps, batch, _ = frames.shape
    count = len(parameters.weights)
    specs = (LayerSpec * count)()
    for index, (weight, bias, state) in enumerate(
        zip(parameters.weights, parameters.biases, states)
    ):
        specs[index] = LayerSpec(
            weight.shape[0],
            weight.shape[1],
            weight.ctypes.data,
            0 if bias is None else bias.ctypes.data,
            0 if bias is None else bias.size,
            parameters.thresholds[index],
            parameters.layer_membrane_fractions[index],
            state.ctypes.data,
            state.size,
        )
    linear = parameters.output_mode == "linear"
    flags = 8 | int(trace) | (2 if binary_inputs else 0) | (4 if linear else 0)
    request = ReplayRequest(
        1,
        flags,
        ctypes.addressof(specs),
        count,
        frames.ctypes.data,
        frames.size,
        steps,
        batch,
        max_working_bytes,
    )
    handle = ctypes.c_void_p()
    result = api.replay(ctypes.byref(request), ctypes.byref(handle))
    if result == -1:
        raise ValueError("native dense IF domain admission refused")
    if result == -2:
        raise MemoryError("native dense IF numeric reservation refused")
    if result == -3:
        raise FloatingPointError("native dense IF arithmetic overflow")
    if result != 0 or handle.value is None:
        raise RuntimeError("native dense IF replay failed to publish an owned result")
    owner = NativeOwner(api, handle)
    output = _view(owner, 0, 0, (batch, parameters.weights[-1].shape[0]))
    final = tuple(
        _view(owner, 1, i, (batch, weight.shape[0])) for i, weight in enumerate(parameters.weights)
    )
    state_trace = (
        tuple(
            _view(owner, 2, i, (steps, batch, weight.shape[0]))
            for i, weight in enumerate(parameters.weights)
        )
        if trace
        else ()
    )
    spike_trace = (
        tuple(
            _view(owner, 3, i, (steps, batch, parameters.weights[i].shape[0]))
            for i in range(count - int(linear))
        )
        if trace
        else ()
    )
    return IFReplayResult(output, final, state_trace, spike_trace)
