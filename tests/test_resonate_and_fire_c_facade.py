# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Resonate-and-fire native C facade contracts

"""Verify native runner availability and Go/Mojo C facade rejection boundaries."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sc_neurocore.accel import resonate_and_fire as backends
from sc_neurocore.accel.go import resonate_and_fire as go_backend
from sc_neurocore.accel.mojo import resonate_and_fire as mojo_backend
from tests.resonate_and_fire_accel_dispatch_support import _PARAMETERS


def test_native_runner_rechecks_rust_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "_engine_simulate", None)
    with pytest.raises(RuntimeError, match="Rust resonate-and-fire backend is unavailable"):
        backends._native_runner("rust")


@pytest.mark.parametrize("module", (go_backend, mojo_backend))
def test_c_facade_input_shape_and_missing_library_fail_closed(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        module.simulate_resonate_and_fire(*_PARAMETERS, np.zeros((2, 2)))
    monkeypatch.setattr(module, "_lib", None)
    with pytest.raises(ImportError, match="libresonate_and_fire.so not built"):
        module.simulate_resonate_and_fire(*_PARAMETERS, np.asarray([1.5]))


@pytest.mark.parametrize("module", (go_backend, mojo_backend))
def test_c_facade_step_bound_precedes_contiguous_copy(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = np.broadcast_to(np.asarray([1.0]), ((1 << 31),))

    def unexpected_copy(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("oversized logical input reached contiguous allocation")

    monkeypatch.setattr(module.np, "ascontiguousarray", unexpected_copy)
    with pytest.raises(ValueError, match="signed-32-bit step limit"):
        module.simulate_resonate_and_fire(*_PARAMETERS, oversized)


@pytest.mark.parametrize("module", (go_backend, mojo_backend))
def test_c_facade_library_probe_handles_loader_failure(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_load(_path: str) -> None:
        raise OSError("shared library unavailable")

    monkeypatch.setattr(module.ctypes, "CDLL", fail_to_load)
    assert module._load_library() == (None, False)


@pytest.mark.parametrize(
    ("status", "exception", "message"),
    (
        (1, RuntimeError, "code 1"),
        (2, ValueError, "configuration"),
        (3, ValueError, "current"),
        (4, FloatingPointError, "candidate"),
    ),
)
@pytest.mark.parametrize("module", (go_backend, mojo_backend))
def test_c_facade_maps_each_native_status(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    exception: type[Exception],
    message: str,
) -> None:
    fake = SimpleNamespace(resonate_and_fire_simulate_c=lambda *_args: status)
    monkeypatch.setattr(module, "_lib", fake)
    with pytest.raises(exception, match=message):
        module.simulate_resonate_and_fire(*_PARAMETERS, np.asarray([1.5]))
