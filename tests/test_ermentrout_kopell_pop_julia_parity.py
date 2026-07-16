# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python/Julia MPR equation-(12) parity

"""Compare the Julia batch with the Python golden."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel.ermentrout_kopell_pop import simulate_python
from sc_neurocore.accel.julia.neurons import (
    _HAS_JULIA_NEURONS,
    _ensure_ermentrout_kopell_pop_loaded,
    is_julia_error,
    simulate_ermentrout_kopell_pop,
)

_PARAMETERS = (0.13, -1.7, 1.3, 0.8, -4.2, 12.5, 0.004)


def _drive(steps: int) -> npt.NDArray[np.float64]:
    index = np.arange(steps, dtype=np.float64)
    return 1.5 + 0.5 * np.sin(index * 0.037)


def test_julia_runtime_and_kernel_are_available() -> None:
    assert _HAS_JULIA_NEURONS
    assert _ensure_ermentrout_kopell_pop_loaded() is not None


@pytest.mark.parametrize("steps", (0, 1, 128, 1024))
def test_julia_complete_mapping_matches_python(steps: int) -> None:
    expected = simulate_python(*_PARAMETERS, _drive(steps))
    actual = simulate_ermentrout_kopell_pop(*_PARAMETERS, _drive(steps))
    for key in ("r", "v"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=1.0e-12)


def test_julia_rejects_nonfinite_drive_before_dispatch() -> None:
    with pytest.raises(ValueError, match="finite"):
        simulate_ermentrout_kopell_pop(*_PARAMETERS, [1.5, np.nan])


def _direct_call(
    drive: npt.NDArray[np.float64],
    r_out: npt.NDArray[np.generic],
    v_out: npt.NDArray[np.generic],
    parameters: tuple[float, ...] = _PARAMETERS,
) -> object:
    module = _ensure_ermentrout_kopell_pop_loaded()
    return module.simulate_ermentrout_kopell_pop_b(
        *parameters,
        drive,
        r_out,
        v_out,
    )


def _assert_typed_julia_error(
    call: Callable[[], object],
    predicate_name: str,
) -> None:
    module = _ensure_ermentrout_kopell_pop_loaded()
    with pytest.raises(Exception) as raised:
        call()
    assert is_julia_error(raised.value)
    predicate = getattr(module, predicate_name)
    assert predicate(cast(Any, raised.value).exception)


def test_julia_rejects_non_float_read_only_and_strided_outputs_atomically() -> None:
    drive = _drive(4)

    integer = np.full(4, -999, dtype=np.int64)
    peer = np.full(4, -999.0, dtype=np.float64)
    _assert_typed_julia_error(
        lambda: _direct_call(drive, integer, peer),
        "is_configuration_error",
    )
    np.testing.assert_array_equal(integer, np.full(4, -999, dtype=np.int64))
    np.testing.assert_array_equal(peer, np.full(4, -999.0))

    read_only = np.full(4, -999.0, dtype=np.float64)
    read_only.setflags(write=False)
    peer.fill(-999.0)
    _assert_typed_julia_error(
        lambda: _direct_call(drive, read_only, peer),
        "is_configuration_error",
    )
    np.testing.assert_array_equal(read_only, np.full(4, -999.0))
    np.testing.assert_array_equal(peer, np.full(4, -999.0))

    r_base = np.full(8, -999.0, dtype=np.float64)
    v_base = np.full(8, -999.0, dtype=np.float64)
    _assert_typed_julia_error(
        lambda: _direct_call(drive, r_base[::2], v_base[::2]),
        "is_configuration_error",
    )
    np.testing.assert_array_equal(r_base, np.full(8, -999.0))
    np.testing.assert_array_equal(v_base, np.full(8, -999.0))


def test_julia_rejects_output_and_input_aliasing_atomically() -> None:
    drive = _drive(4)

    identical = np.full(4, -999.0, dtype=np.float64)
    _assert_typed_julia_error(
        lambda: _direct_call(drive, identical, identical),
        "is_configuration_error",
    )
    np.testing.assert_array_equal(identical, np.full(4, -999.0))

    output_base = np.full(5, -999.0, dtype=np.float64)
    _assert_typed_julia_error(
        lambda: _direct_call(drive, output_base[:4], output_base[1:]),
        "is_configuration_error",
    )
    np.testing.assert_array_equal(output_base, np.full(5, -999.0))

    shared = np.empty(5, dtype=np.float64)
    shared[:4] = drive
    shared[4] = -999.0
    peer = np.full(4, -999.0, dtype=np.float64)
    before = shared.copy()
    _assert_typed_julia_error(
        lambda: _direct_call(shared[:4], shared[1:], peer),
        "is_configuration_error",
    )
    np.testing.assert_array_equal(shared, before)
    np.testing.assert_array_equal(peer, np.full(4, -999.0))


def test_julia_late_candidate_failure_leaves_both_outputs_unchanged() -> None:
    drive = np.full(8, -10.0, dtype=np.float64)
    r_out = np.full(8, -999.0, dtype=np.float64)
    v_out = np.full(8, -999.0, dtype=np.float64)
    parameters = (0.01, -10.0, 0.1, 0.1, -10.0, -20.0, 0.2)

    _assert_typed_julia_error(
        lambda: _direct_call(drive, r_out, v_out, parameters),
        "is_candidate_error",
    )
    np.testing.assert_array_equal(r_out, np.full(8, -999.0))
    np.testing.assert_array_equal(v_out, np.full(8, -999.0))


def test_julia_facade_maps_only_typed_mpr_failures() -> None:
    with pytest.raises(ValueError):
        simulate_ermentrout_kopell_pop(*_PARAMETERS[:-1], -_PARAMETERS[-1], [0.0])
    with pytest.raises(FloatingPointError):
        simulate_ermentrout_kopell_pop(
            1.0,
            -100.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.1,
            [0.0],
        )
