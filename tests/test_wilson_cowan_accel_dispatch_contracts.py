# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson-Cowan acceleration dispatcher contracts

"""Contracts for Wilson-Cowan Go, Mojo, and Julia acceleration facades."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from types import ModuleType
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel.go import wilson_cowan as go_wilson
from sc_neurocore.accel.julia import neurons as julia_neurons
from sc_neurocore.accel.mojo import wilson_cowan as mojo_wilson

WilsonResult = dict[str, Any]
WilsonDispatcher = Callable[
    [
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        npt.ArrayLike,
    ],
    WilsonResult,
]


class WilsonCShim(Protocol):
    """Callable C-shim protocol for Wilson-Cowan ctypes facades."""

    def __call__(self, *args: object) -> int:
        """Return the simulated C ABI status code."""


class FakeCShim:
    """C-shim stub that returns a configured status code."""

    argtypes: list[object] = []
    restype: object = ctypes.c_int

    def __init__(self, return_code: int) -> None:
        """Store the return code used by the fake C ABI call."""
        self._return_code = return_code

    def __call__(self, *args: object) -> int:
        """Return the configured C ABI status code."""
        return self._return_code


class FakeWilsonLib:
    """Shared-library stub exposing the Wilson-Cowan C symbol."""

    def __init__(self, return_code: int = 0) -> None:
        """Initialise the fake library with a C-shim return code."""
        self.wilson_cowan_simulate_c: WilsonCShim = FakeCShim(return_code)


class FakeJuliaWilsonModule:
    """Julia module stub exposing the Wilson-Cowan batch function."""

    def simulate_wilson_cowan_b(
        self,
        e_init: float,
        i_init: float,
        w_ee: float,
        w_ei: float,
        w_ie: float,
        w_ii: float,
        tau_e: float,
        tau_i: float,
        a: float,
        theta: float,
        dt: float,
        ext_input: npt.NDArray[np.float64],
        e_out: npt.NDArray[np.float64],
        i_out: npt.NDArray[np.float64],
    ) -> tuple[float, float]:
        """Populate output buffers with a deterministic fake trajectory."""
        del w_ee, w_ei, w_ie, w_ii, tau_e, tau_i, a, theta, dt
        e_out[:] = e_init + ext_input
        i_out[:] = i_init
        return float(e_out[-1]) if e_out.size else e_init, i_init


def _call_dispatcher(
    dispatcher: WilsonDispatcher,
    ext_input: npt.ArrayLike,
) -> WilsonResult:
    """Invoke a Wilson-Cowan acceleration dispatcher with canonical parameters."""
    return dispatcher(
        0.1,
        0.05,
        10.0,
        6.0,
        10.0,
        1.0,
        1.0,
        2.0,
        1.2,
        4.0,
        0.1,
        ext_input,
    )


@pytest.mark.parametrize(
    "module,dispatcher",
    [
        (go_wilson, go_wilson.simulate_wilson_cowan),
        (mojo_wilson, mojo_wilson.simulate_wilson_cowan),
    ],
)
def test_ctypes_wilson_cowan_rejects_non_1d_external_input(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    dispatcher: WilsonDispatcher,
) -> None:
    """Go and Mojo ctypes facades reject matrix-valued external input."""
    monkeypatch.setattr(module, "_lib", FakeWilsonLib())

    with pytest.raises(ValueError, match="ext_input must be one-dimensional"):
        _call_dispatcher(dispatcher, np.zeros((2, 2), dtype=np.float64))


def test_julia_wilson_cowan_rejects_non_1d_external_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Julia facade rejects matrix-valued external input before backend calls."""
    monkeypatch.setattr(
        julia_neurons,
        "_ensure_wilson_cowan_loaded",
        lambda: FakeJuliaWilsonModule(),
    )

    with pytest.raises(ValueError, match="ext_input must be one-dimensional"):
        _call_dispatcher(
            julia_neurons.simulate_wilson_cowan,
            np.zeros((2, 2), dtype=np.float64),
        )


@pytest.mark.parametrize(
    "module,dispatcher,error_message",
    [
        (go_wilson, go_wilson.simulate_wilson_cowan, "wilson_cowan_simulate_c"),
        (mojo_wilson, mojo_wilson.simulate_wilson_cowan, "Mojo wilson_cowan"),
    ],
)
def test_ctypes_wilson_cowan_nonzero_return_code_raises(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    dispatcher: WilsonDispatcher,
    error_message: str,
) -> None:
    """Go and Mojo facades report non-zero C ABI return codes."""
    monkeypatch.setattr(module, "_lib", FakeWilsonLib(return_code=42))

    with pytest.raises(RuntimeError, match=f"{error_message}.*42"):
        _call_dispatcher(dispatcher, np.zeros(3, dtype=np.float64))


@pytest.mark.parametrize(
    "module,dispatcher",
    [
        (go_wilson, go_wilson.simulate_wilson_cowan),
        (mojo_wilson, mojo_wilson.simulate_wilson_cowan),
    ],
)
def test_ctypes_wilson_cowan_success_returns_one_dimensional_buffers(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    dispatcher: WilsonDispatcher,
) -> None:
    """Go and Mojo facades return one-dimensional output buffers on success."""
    monkeypatch.setattr(module, "_lib", FakeWilsonLib(return_code=0))

    out = _call_dispatcher(dispatcher, np.array([0.0, 1.0, 2.0], dtype=np.float64))

    assert isinstance(out["e"], np.ndarray)
    assert isinstance(out["i"], np.ndarray)
    assert out["e"].shape == (3,)
    assert out["i"].shape == (3,)
    assert out["e"].dtype == np.dtype(np.float64)
    assert out["i"].dtype == np.dtype(np.float64)
    assert out["e_final"] == 0.0
    assert out["i_final"] == 0.0


@pytest.mark.parametrize(
    "module,dispatcher",
    [
        (go_wilson, go_wilson.simulate_wilson_cowan),
        (mojo_wilson, mojo_wilson.simulate_wilson_cowan),
    ],
)
def test_ctypes_wilson_cowan_unavailable_library_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    dispatcher: WilsonDispatcher,
) -> None:
    """Go and Mojo facades raise an install hint when no shared library exists."""
    monkeypatch.setattr(module, "_lib", None)

    with pytest.raises(ImportError, match="libwilson_cowan.so not built"):
        _call_dispatcher(dispatcher, np.zeros(3, dtype=np.float64))


@pytest.mark.parametrize("module", [go_wilson, mojo_wilson])
def test_ctypes_wilson_cowan_loader_failure_sets_unavailable_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> None:
    """Shared-library load failures return a missing-library sentinel pair."""

    def fail_cdll(_path: str) -> ctypes.CDLL:
        """Raise the same error type used by a missing shared library."""
        raise OSError("shared library not available")

    monkeypatch.setattr(ctypes, "CDLL", fail_cdll)

    lib, has_library = module._load_library()

    assert lib is None
    assert has_library is False
