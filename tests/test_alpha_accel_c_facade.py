# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Alpha accelerator C-facade contracts

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sc_neurocore.accel.go import alpha as go_backend
from sc_neurocore.accel.mojo import alpha as mojo_backend
from tests.alpha_accel_dispatch_support import PARAMETERS


@pytest.mark.parametrize("module", (go_backend, mojo_backend))
def test_c_facade_input_shape_and_missing_library_fail_closed(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        module.simulate_alpha(*PARAMETERS, np.zeros((2, 2)))
    monkeypatch.setattr(module, "_lib", None)
    with pytest.raises(ImportError, match="libalpha.so not built"):
        module.simulate_alpha(*PARAMETERS, np.asarray([1.5]))


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
        module.simulate_alpha(*PARAMETERS, oversized)


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
    fake = SimpleNamespace(alpha_simulate_c=lambda *_args: status)
    monkeypatch.setattr(module, "_lib", fake)
    with pytest.raises(exception, match=message):
        module.simulate_alpha(*PARAMETERS, np.asarray([2.0, 2.5]), np.asarray([0.5, 0.75]))
