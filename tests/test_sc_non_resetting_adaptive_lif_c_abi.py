# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC non-resetting adaptive LIF C ABI refusal and caller-buffer contracts

"""Exercise real Go/Mojo exported functions and caller-owned output buffers."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from tests.test_sc_non_resetting_adaptive_lif_engine_binding_configuration import (
    FIELDS,
    INVALID_CONFIGURATIONS,
    _parameters,
)

ALL_INVALID_CONFIGURATIONS = [
    *INVALID_CONFIGURATIONS,
    *[{field: value} for field in FIELDS for value in (float("nan"), float("inf"), -float("inf"))],
]

_REPOSITORY = Path(__file__).resolve().parents[1]


def _call(
    backend: str,
    parameters: dict[str, float],
    count: int,
    current: float | npt.NDArray[np.float64] = 30.0,
    null_buffer: int | None = None,
) -> tuple[int, list[npt.NDArray[np.float64]], npt.NDArray[np.int64]]:
    """Invoke the exported C ABI with live, writable, typed caller buffers."""
    library = ctypes.CDLL(
        str(
            _REPOSITORY
            / f"src/sc_neurocore/accel/{backend}/sc_non_resetting_adaptive_lif/libsc_non_resetting_adaptive_lif.so"
        )
    )
    native = library.sc_non_resetting_adaptive_lif_simulate_c
    native.restype = ctypes.c_int if backend == "go" else ctypes.c_ssize_t
    function = cast(Callable[..., int], native)
    floating = [np.full(max(count, 1), current)] + [
        np.full(max(count, 1), -777.0) for _ in range(2)
    ]
    events = np.full(max(count, 1), -777, dtype=np.int64)
    finals = [np.full(1, -777.0) for _ in range(2)]
    addresses = [array.ctypes.data for array in [*floating, events, *finals]]
    if null_buffer is not None:
        addresses[null_buffer] = 0
    size = ctypes.c_int(count) if backend == "go" else ctypes.c_ssize_t(count)
    pointers = (
        [ctypes.c_void_p(address) for address in addresses]
        if backend == "go"
        else [ctypes.c_ssize_t(address) for address in addresses]
    )
    status = function(
        size, *(ctypes.c_double(v) for v in _parameters(parameters).values()), *pointers
    )
    return status, [*floating, *finals], events


@pytest.mark.parametrize("parameters", ALL_INVALID_CONFIGURATIONS)
@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("backend", ["go", "mojo"])
def test_invalid_configuration_cannot_write_caller_buffers(
    parameters: dict[str, float], count: int, backend: str
) -> None:
    """Configuration refusal precedes every output write, including zero-step finals."""
    status, floating, events = _call(backend, parameters, count)
    assert status == 2
    np.testing.assert_array_equal(floating[0], np.full(max(count, 1), 30.0))
    for array in floating[1:]:
        np.testing.assert_array_equal(array, np.full(array.size, -777.0))
    np.testing.assert_array_equal(events, np.full(events.size, -777))
    recovered, _, _ = _call(backend, {}, 1)
    assert recovered == 0


@pytest.mark.parametrize("backend", ["go", "mojo"])
@pytest.mark.parametrize("current", [float("nan"), float("inf"), -float("inf")])
def test_first_transition_refusal_preserves_outputs(backend: str, current: float) -> None:
    """A refused first sample leaves trace and final-state buffers untouched."""
    status, floating, events = _call(backend, {}, 1, current)
    assert status == 2
    for array in floating[1:]:
        np.testing.assert_array_equal(array, np.full(array.size, -777.0))
    assert events[0] == -777


@pytest.mark.parametrize("backend", ["go", "mojo"])
def test_valid_zero_step_c_abi_writes_only_final_state(backend: str) -> None:
    """A valid empty batch commits exact initial finals and retains trace sentinels."""
    status, floating, events = _call(backend, {}, 0)
    assert status == 0
    assert (floating[-2][0], floating[-1][0]) == (-65.0, -50.0)
    assert floating[1][0] == floating[2][0] == -777.0
    assert events[0] == -777


@pytest.mark.parametrize("backend", ["go", "mojo"])
@pytest.mark.parametrize("buffer", range(6))
@pytest.mark.parametrize("count", [0, 1])
def test_null_caller_pointer_refuses_before_any_output(
    backend: str, buffer: int, count: int
) -> None:
    """Every absent pointer refuses empty and nonempty calls without writing finals."""
    status, floating, events = _call(backend, {}, count, null_buffer=buffer)
    assert status == 1
    for array in floating[1:]:
        np.testing.assert_array_equal(array, np.full(array.size, -777.0))
    assert events[0] == -777


@pytest.mark.parametrize("backend", ["go", "mojo"])
def test_negative_count_refuses_without_writes(backend: str) -> None:
    """A negative length cannot reach a buffer view or any state commit."""
    status, floating, events = _call(backend, {}, -1)
    assert status == 1
    for array in floating[1:]:
        np.testing.assert_array_equal(array, np.full(array.size, -777.0))
    assert events[0] == -777


@pytest.mark.parametrize("backend", ["go", "mojo"])
@pytest.mark.parametrize("current", [float("nan"), float("inf"), -float("inf")])
def test_late_refusal_retains_valid_prefix_without_writing_finals(
    backend: str, current: float
) -> None:
    """A late refusal exposes the accepted prefix and preserves uncommitted buffers."""
    from sc_neurocore.neurons.models.sc_non_resetting_adaptive_lif import (
        SCNonResettingAdaptiveLIFNeuron,
    )

    drive = np.array([20.0, current, 10.0])
    status, floating, events = _call(backend, {}, 3, current=drive)
    reference = SCNonResettingAdaptiveLIFNeuron()
    event = reference.step(20.0)
    assert status == 2 and events[0] == event
    np.testing.assert_allclose(
        [floating[1][0], floating[2][0]], [reference.v, reference.theta], rtol=0, atol=2e-12
    )
    for output in floating[1:3]:
        np.testing.assert_array_equal(output[1:], [-777.0, -777.0])
    np.testing.assert_array_equal(events[1:], [-777, -777])
    assert floating[-2][0] == floating[-1][0] == -777.0
    np.testing.assert_array_equal(floating[0], drive)


@pytest.mark.parametrize("backend", ["go", "mojo"])
@pytest.mark.parametrize(
    ("parameters", "current"),
    [
        ({"r_m": 1e308}, 20.0),
        ({"v": 1.5e308, "theta": 1e308, "theta_rest": 1e308, "delta_theta": 1e308}, 0.0),
    ],
)
def test_finite_overflow_retains_caller_outputs(
    backend: str, parameters: dict[str, float], current: float
) -> None:
    """A finite candidate overflow refuses before any first-sample output write."""
    status, floating, events = _call(backend, parameters, 1, current=current)
    assert status == 2
    for output in floating[1:]:
        assert output[0] == -777.0
    assert events[0] == -777
