# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dense IF replay backend selection

"""Dispatch public dense IF replay to an available native library or the NumPy floor."""

import os
from collections.abc import Sequence
from typing import Literal, cast

import numpy.typing as npt

from .if_benchmark_order import measured_order
from .if_julia import load_julia
from .if_native import load_native, replay_native
from .if_native_types import NativeAPI
from .if_parameters import IFParameters
from .if_replay import IFReplayResult, replay_dense_if

ReplayBackend = Literal["auto", "numpy", "rust", "go", "mojo", "julia"]


def replay_backend(
    parameters: IFParameters,
    inputs: npt.ArrayLike,
    initial_state: Sequence[npt.ArrayLike] | None,
    trace: bool,
    binary_inputs: bool,
    max_working_bytes: int,
    backend: ReplayBackend,
) -> IFReplayResult:
    """Select an explicitly configured native replay or the always-available floor.

    Parameters
    ----------
    parameters : IFParameters
        Owned finite coefficients and response semantics.
    inputs : array_like
        Explicit replay frames.
    initial_state : sequence of array_like or None
        Optional continuation states copied before execution.
    trace : bool
        Retain complete post-step state/event trajectories.
    binary_inputs : bool
        Require exact zero/one events when True.
    max_working_bytes : int
        Positive numeric replay buffer limit.
    backend : {'auto', 'numpy', 'rust', 'go', 'mojo', 'julia'}
        Auto uses a validated configured comparison, or the static native order, before NumPy.

    Returns
    -------
    IFReplayResult
        Owned responses and complete requested trajectories.

    Raises
    ------
    ValueError
        Unknown backend or invalid replay domains.
    RuntimeError
        Requested native library absent or incompatible.
    """
    native = select_native(backend)
    if native is not None:
        return replay_native(
            native,
            parameters,
            inputs,
            initial_state,
            trace,
            binary_inputs,
            max_working_bytes,
        )
    return replay_dense_if(
        parameters,
        inputs,
        initial_state=initial_state,
        trace=trace,
        binary_inputs=binary_inputs,
        max_working_bytes=max_working_bytes,
    )


def resolve_replay_backend(
    backend: ReplayBackend,
) -> Literal["numpy", "rust", "go", "mojo", "julia"]:
    """Name the runtime a replay request executes on, without loading it.

    Parameters
    ----------
    backend : {'auto', 'numpy', 'rust', 'go', 'mojo', 'julia'}
        Auto orders configured native providers by a validated optional comparison;
        without one, Rust then Go then Mojo then Julia. NumPy always uses the floor.

    Returns
    -------
    {'numpy', 'rust', 'go', 'mojo', 'julia'}
        The explicit name, or the provider auto resolves to under the current
        configuration. Requesting that name explicitly selects the same runtime.

    Raises
    ------
    ValueError
        Backend name is unsupported.
    RuntimeError
        The Julia opt-in is malformed, or the measured comparison is invalid.
    """
    if backend not in ("auto", "numpy", "rust", "go", "mojo", "julia"):
        raise ValueError("unsupported dense IF backend")
    if backend != "auto":
        return backend
    for name in measured_order():
        if name == "julia":
            if os.environ.get("SC_NEUROCORE_IF_JULIA_ENABLED") == "1":
                return "julia"
        elif os.environ.get(f"SC_NEUROCORE_IF_{name.upper()}_LIB"):
            return cast(Literal["rust", "go", "mojo"], name)
    enabled = os.environ.get("SC_NEUROCORE_IF_JULIA_ENABLED", "0")
    if enabled not in ("0", "1"):
        raise RuntimeError("Julia dense IF opt-in must be 0 or 1")
    return "numpy"


def select_native(backend: ReplayBackend) -> NativeAPI | None:
    """Resolve the requested native runtime, including for empty input batches.

    Parameters
    ----------
    backend : {'auto', 'numpy', 'rust', 'go', 'mojo', 'julia'}
        Auto orders configured native providers by a validated optional comparison;
        without one, Rust then Go then Mojo then Julia. NumPy always uses the floor.

    Returns
    -------
    NativeAPI or None
        Loaded native ownership API or the selected NumPy floor.

    Raises
    ------
    ValueError
        Backend name is unsupported.
    RuntimeError
        Explicit native configuration is absent or its library cannot load.
    """
    name = resolve_replay_backend(backend)
    if name == "numpy":
        return None
    if name == "julia":
        if os.environ.get("SC_NEUROCORE_IF_JULIA_ENABLED") != "1":
            raise RuntimeError("Julia dense IF backend requires SC_NEUROCORE_IF_JULIA_ENABLED=1")
        return load_julia()
    variable = f"SC_NEUROCORE_IF_{name.upper()}_LIB"
    configured = os.environ.get(variable)
    if not configured:
        raise RuntimeError(f"{name} dense IF backend requires {variable}")
    return load_native(configured)
