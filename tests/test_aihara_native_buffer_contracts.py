# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Aihara native buffer ownership contracts

"""Exercise the real Go/Mojo C ABI with caller-owned contiguous buffers."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import aihara_map
from tests.native_lane_requirement import require_native_lane

_ROOT = Path(__file__).resolve().parents[1]
_PARAMETERS = (0.1, 0.7, 1.0, 0.3968, 0.01)
_OVERLAPS = tuple(
    (left, right, offset)
    for left, right in combinations(range(7), 2)
    for offset in ((-1, 0, 1) if right < 4 else (0, 1) if left < 4 else (0,))
)
NativeCall = Callable[..., int]


@pytest.fixture(params=("go", "mojo"))
def backend_name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def native_call(backend_name: str) -> NativeCall:
    """Load the same exported C symbol used by the production dispatcher."""
    backend = backend_name
    require_native_lane(aihara_map, backend, subject="Aihara buffer ownership")
    library = ctypes.CDLL(
        str(_ROOT / "src/sc_neurocore/accel" / backend / "aihara_map/libaihara_map.so")
    )
    function = library.aihara_map_simulate_c
    function.argtypes = [ctypes.c_int32, *([ctypes.c_double] * 5), *([ctypes.c_void_p] * 7)]
    function.restype = ctypes.c_int32
    return cast(NativeCall, function)


def _buffers(steps: int) -> list[npt.NDArray[np.float64]]:
    return [np.full(max(steps, 1) + 1, -123.0) for _ in range(7)]


@pytest.mark.parametrize(("left", "right", "offset"), _OVERLAPS)
def test_overlapping_regions_are_rejected_without_writes(
    native_call: NativeCall, left: int, right: int, offset: int
) -> None:
    """Every pair must reject full or partial aliasing, including input/finals."""
    buffers = _buffers(3)
    buffers[0][:] = 0.0
    pointers = [array.ctypes.data for array in buffers]
    if offset < 0:
        pointers[left] = pointers[right] + 8
    else:
        pointers[right] = pointers[left] + offset * 8
    before = [array.copy() for array in buffers]
    assert native_call(3, *_PARAMETERS, *pointers) == 1
    for actual, expected in zip(buffers, before, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(("left", "right"), tuple(combinations(range(4, 7), 2)))
def test_empty_batch_still_requires_distinct_final_outputs(
    native_call: NativeCall, left: int, right: int
) -> None:
    buffers = _buffers(0)
    pointers = [0, 0, 0, 0, *(array.ctypes.data for array in buffers[4:])]
    pointers[right] = pointers[left]
    assert native_call(0, *_PARAMETERS, *pointers) == 1
    assert all(np.all(array == -123.0) for array in buffers)


@pytest.mark.parametrize("steps", (0, 1, 64))
def test_adjacent_disjoint_storage_preserves_complete_receipt(
    native_call: NativeCall, backend_name: str, steps: int
) -> None:
    """One backing allocation is valid when the active ranges do not overlap."""
    storage = np.full(4 * steps + 3, -123.0)
    views = [storage[index * steps : (index + 1) * steps] for index in range(4)]
    views.extend(storage[4 * steps + index : 4 * steps + index + 1] for index in range(3))
    views[0][:] = 0.0
    expected = aihara_map.simulate_aihara_map(current=views[0].copy(), backend=backend_name)
    reference = aihara_map.simulate_aihara_map(current=views[0].copy(), backend="python")
    assert native_call(steps, *_PARAMETERS, *(array.ctypes.data for array in views)) == 0
    np.testing.assert_array_equal(views[0], np.zeros(steps))
    for actual, key in zip(views[1:4], ("y", "x", "spikes"), strict=True):
        np.testing.assert_array_equal(actual, expected[key])
        if key == "spikes":
            np.testing.assert_array_equal(actual, reference[key])
        else:
            np.testing.assert_allclose(
                actual, reference[key], rtol=0.0, atol=aihara_map.PARITY_ATOL[backend_name]
            )
    for actual, key in zip(views[4:], ("y_final", "x_final", "spike_count"), strict=True):
        assert actual[0] == expected[key]


@pytest.mark.parametrize("fault", ("input", "candidate"))
def test_numerical_failure_preserves_all_output_buffers(
    native_call: NativeCall, fault: str
) -> None:
    buffers = _buffers(3)
    buffers[0][:] = (0.0, 0.0, np.nan, 0.0) if fault == "input" else 1.7e308
    parameters = _PARAMETERS if fault == "input" else (1.7e308, 0.99, 1.0, 0.3968, 0.01)
    before = [array.copy() for array in buffers]
    assert native_call(3, *parameters, *(array.ctypes.data for array in buffers)) == (
        3 if fault == "input" else 4
    )
    for actual, expected in zip(buffers, before, strict=True):
        np.testing.assert_array_equal(actual, expected)
