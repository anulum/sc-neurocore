# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python/Julia adaptive-threshold parity

"""Compare Julia exact-relaxation execution and typed buffer failures with Python."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import adaptive_threshold_if as backends

_PARAMETERS = (-63.5, -52.5, -68.0, -67.0, -49.0, 4.5, 8.0, 42.0, 0.05)
_FAILURE_PARAMETERS = (-1.0e308, -45.0, -65.0, -65.0, -50.0, 5.0, 10.0, 50.0, 0.1)


def _drive(steps: int) -> npt.NDArray[np.float64]:
    index = np.arange(steps, dtype=np.float64)
    return 22.0 + 6.0 * np.sin(index * 0.037)


def _module() -> Any:
    return backends._ensure_julia_loaded()


def test_julia_runtime_and_kernel_are_available() -> None:
    assert backends.backend_available("julia")
    assert _module() is not None


@pytest.mark.parametrize("steps", (0, 1, 128, 1024))
def test_julia_complete_mapping_matches_python(steps: int) -> None:
    expected = backends.simulate_python(*_PARAMETERS, _drive(steps))
    actual = backends.simulate_adaptive_threshold_if(
        *_PARAMETERS,
        _drive(steps),
        backend="julia",
    )
    for key in ("v", "theta", "spikes"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=1.0e-12)
    assert actual["spike_count"] == expected["spike_count"]


def test_julia_rejects_nonfinite_drive_before_dispatch() -> None:
    with pytest.raises(ValueError, match="finite"):
        backends.simulate_adaptive_threshold_if(
            *_PARAMETERS,
            [1.5, np.nan],
            backend="julia",
        )


def _direct_call(
    drive: npt.NDArray[np.float64],
    v_out: npt.NDArray[np.generic],
    theta_out: npt.NDArray[np.generic],
    spikes_out: npt.NDArray[np.generic],
    parameters: tuple[float, ...] = _PARAMETERS,
) -> object:
    return _module().simulate_adaptive_threshold_if_b(
        *parameters,
        drive,
        v_out,
        theta_out,
        spikes_out,
    )


def _assert_typed_julia_error(
    call: Callable[[], object],
    predicate_name: str,
) -> None:
    module = _module()
    with pytest.raises(Exception) as raised:
        call()
    assert raised.value.__class__.__name__ == "JuliaError"
    predicate = getattr(module, predicate_name)
    assert predicate(cast(Any, raised.value).exception)


def test_julia_rejects_nonfloat_readonly_and_strided_outputs_atomically() -> None:
    drive = _drive(4)
    integer = np.full(4, -999, dtype=np.int64)
    peer_theta = np.full(4, -999.0, dtype=np.float64)
    peer_spikes = np.full(4, -999.0, dtype=np.float64)
    _assert_typed_julia_error(
        lambda: _direct_call(drive, integer, peer_theta, peer_spikes),
        "is_configuration_error",
    )
    np.testing.assert_array_equal(integer, np.full(4, -999, dtype=np.int64))
    np.testing.assert_array_equal(peer_theta, np.full(4, -999.0))
    np.testing.assert_array_equal(peer_spikes, np.full(4, -999.0))

    read_only = np.full(4, -999.0, dtype=np.float64)
    read_only.setflags(write=False)
    _assert_typed_julia_error(
        lambda: _direct_call(drive, read_only, peer_theta, peer_spikes),
        "is_configuration_error",
    )
    np.testing.assert_array_equal(read_only, np.full(4, -999.0))

    v_base = np.full(8, -999.0, dtype=np.float64)
    theta_base = np.full(8, -999.0, dtype=np.float64)
    spike_base = np.full(8, -999.0, dtype=np.float64)
    _assert_typed_julia_error(
        lambda: _direct_call(
            drive,
            v_base[::2],
            theta_base[::2],
            spike_base[::2],
        ),
        "is_configuration_error",
    )
    np.testing.assert_array_equal(v_base, np.full(8, -999.0))
    np.testing.assert_array_equal(theta_base, np.full(8, -999.0))
    np.testing.assert_array_equal(spike_base, np.full(8, -999.0))


def test_julia_rejects_every_obvious_alias_class_atomically() -> None:
    drive = _drive(4)
    identical = np.full(4, -999.0, dtype=np.float64)
    peer = np.full(4, -999.0, dtype=np.float64)
    _assert_typed_julia_error(
        lambda: _direct_call(drive, identical, identical, peer),
        "is_configuration_error",
    )
    np.testing.assert_array_equal(identical, np.full(4, -999.0))

    output_base = np.full(6, -999.0, dtype=np.float64)
    _assert_typed_julia_error(
        lambda: _direct_call(
            drive,
            output_base[:4],
            output_base[1:5],
            peer,
        ),
        "is_configuration_error",
    )
    np.testing.assert_array_equal(output_base, np.full(6, -999.0))

    shared = np.empty(5, dtype=np.float64)
    shared[:4] = drive
    shared[4] = -999.0
    before = shared.copy()
    _assert_typed_julia_error(
        lambda: _direct_call(shared[:4], shared[1:], peer, np.full(4, -999.0)),
        "is_configuration_error",
    )
    np.testing.assert_array_equal(shared, before)


def test_julia_late_candidate_failure_leaves_all_outputs_unchanged() -> None:
    drive = np.full(4, 1.0e308, dtype=np.float64)
    v_out = np.full(4, -999.0, dtype=np.float64)
    theta_out = np.full(4, -999.0, dtype=np.float64)
    spikes_out = np.full(4, -999.0, dtype=np.float64)
    _assert_typed_julia_error(
        lambda: _direct_call(drive, v_out, theta_out, spikes_out, _FAILURE_PARAMETERS),
        "is_candidate_error",
    )
    np.testing.assert_array_equal(v_out, np.full(4, -999.0))
    np.testing.assert_array_equal(theta_out, np.full(4, -999.0))
    np.testing.assert_array_equal(spikes_out, np.full(4, -999.0))


def test_julia_facade_maps_only_typed_model_failures() -> None:
    with pytest.raises(ValueError):
        backends.simulate_adaptive_threshold_if(
            *_PARAMETERS[:-1],
            -_PARAMETERS[-1],
            [0.0],
            backend="julia",
        )
    with pytest.raises(FloatingPointError):
        backends.simulate_adaptive_threshold_if(
            *_FAILURE_PARAMETERS,
            [1.0e308],
            backend="julia",
        )
