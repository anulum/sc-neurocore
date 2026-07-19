# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python/Julia alpha-synapse parity

"""Compare Julia exact-flow execution and typed buffer failures with Python."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import alpha as backends

_PARAMETERS = (0.15, 0.08, 0.05, 0.04, 0.03, -0.5, 1.2, 16.0, 4.0, 9.0, 0.5)
_FAILURE_PARAMETERS = (-1.0e308, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 20.0, 5.0, 10.0, 1.0)


def _drive(steps: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    index = np.arange(steps, dtype=np.float64)
    return 2.0 + 0.8 * np.sin(index * 0.037), 0.7 + 0.3 * np.cos(index * 0.021)


def _module() -> Any:
    return backends._ensure_julia_loaded()


def test_julia_runtime_and_kernel_are_available() -> None:
    assert backends.backend_available("julia")
    assert _module() is not None


@pytest.mark.parametrize("steps", (0, 1, 128, 1024))
def test_julia_complete_mapping_matches_python(steps: int) -> None:
    exc, inh = _drive(steps)
    expected = backends.simulate_python(*_PARAMETERS, exc, inh)
    actual = backends.simulate_alpha(*_PARAMETERS, exc, inh, backend="julia")
    for key in ("v", "a_exc", "i_exc", "a_inh", "i_inh", "spikes"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=1.0e-12)
    assert actual["spike_count"] == expected["spike_count"]


def test_julia_rejects_nonfinite_drive_before_dispatch() -> None:
    with pytest.raises(ValueError, match="finite"):
        backends.simulate_alpha(*_PARAMETERS, [1.5, np.nan], [0.1, 0.1], backend="julia")


def _direct_call(
    exc_drive: npt.NDArray[np.float64],
    inh_drive: npt.NDArray[np.float64],
    buffers: dict[str, npt.NDArray[np.generic]],
    parameters: tuple[float, ...] = _PARAMETERS,
) -> object:
    return _module().simulate_alpha_b(
        *parameters,
        exc_drive,
        inh_drive,
        buffers["v"],
        buffers["a_exc"],
        buffers["i_exc"],
        buffers["a_inh"],
        buffers["i_inh"],
        buffers["spikes"],
    )


def _buffers(size: int, fill: float = -999.0) -> dict[str, npt.NDArray[np.float64]]:
    return {
        name: np.full(size, fill, dtype=np.float64)
        for name in ("v", "a_exc", "i_exc", "a_inh", "i_inh", "spikes")
    }


def _assert_typed_julia_error(call: Callable[[], object], predicate_name: str) -> None:
    module = _module()
    with pytest.raises(Exception) as raised:
        call()
    assert raised.value.__class__.__name__ == "JuliaError"
    predicate = getattr(module, predicate_name)
    assert predicate(cast(Any, raised.value).exception)


def test_julia_rejects_nonfloat_readonly_and_strided_outputs_atomically() -> None:
    exc, inh = _drive(4)
    integer = np.full(4, -999, dtype=np.int64)
    peer = _buffers(4)
    _assert_typed_julia_error(
        lambda: _direct_call(exc, inh, {**peer, "v": integer}),
        "is_configuration_error",
    )
    np.testing.assert_array_equal(integer, np.full(4, -999, dtype=np.int64))
    for name, buffer in peer.items():
        np.testing.assert_array_equal(buffer, np.full(4, -999.0))

    read_only = np.full(4, -999.0, dtype=np.float64)
    read_only.setflags(write=False)
    peer2 = _buffers(4)
    _assert_typed_julia_error(
        lambda: _direct_call(exc, inh, {**peer2, "v": read_only}),
        "is_configuration_error",
    )
    np.testing.assert_array_equal(read_only, np.full(4, -999.0))

    base = _buffers(8)
    strided = {name: buffer[::2] for name, buffer in base.items()}
    _assert_typed_julia_error(
        lambda: _direct_call(exc, inh, strided),
        "is_configuration_error",
    )
    for name, buffer in base.items():
        np.testing.assert_array_equal(buffer, np.full(8, -999.0))


def test_julia_rejects_every_obvious_alias_class_atomically() -> None:
    exc, inh = _drive(4)
    identical = np.full(4, -999.0, dtype=np.float64)
    peer = _buffers(4)
    _assert_typed_julia_error(
        lambda: _direct_call(exc, inh, {**peer, "v": identical, "a_exc": identical}),
        "is_configuration_error",
    )
    np.testing.assert_array_equal(identical, np.full(4, -999.0))

    shared = np.empty(5, dtype=np.float64)
    shared[:4] = exc
    shared[4] = -999.0
    before = shared.copy()
    peer2 = _buffers(4)
    _assert_typed_julia_error(
        lambda: _direct_call(shared[:4], inh, {**peer2, "v": shared[1:]}),
        "is_configuration_error",
    )
    np.testing.assert_array_equal(shared, before)


def test_julia_late_candidate_failure_leaves_all_outputs_unchanged() -> None:
    exc = np.full(4, 1.0e308, dtype=np.float64)
    inh = np.zeros(4, dtype=np.float64)
    buffers = _buffers(4)
    _assert_typed_julia_error(
        lambda: _direct_call(exc, inh, buffers, _FAILURE_PARAMETERS),
        "is_candidate_error",
    )
    for name, buffer in buffers.items():
        np.testing.assert_array_equal(buffer, np.full(4, -999.0))


def test_julia_facade_maps_only_typed_model_failures() -> None:
    with pytest.raises(ValueError):
        backends.simulate_alpha(*_PARAMETERS[:-1], -_PARAMETERS[-1], [0.0], [0.0], backend="julia")
    with pytest.raises(FloatingPointError):
        backends.simulate_alpha(*_FAILURE_PARAMETERS, [1.0e308], [0.0], backend="julia")
