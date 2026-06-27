# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wong-Wang acceleration dispatcher contracts

"""Contracts for Wong-Wang Go, Mojo, and Julia acceleration facades."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from types import ModuleType
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel.go import wong_wang as go_wong
from sc_neurocore.accel.julia import neurons as julia_neurons
from sc_neurocore.accel.mojo import wong_wang as mojo_wong

WongResult = dict[str, Any]
WongDispatcher = Callable[
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
        npt.ArrayLike,
        npt.ArrayLike,
        npt.ArrayLike,
    ],
    WongResult,
]


class WongWangCShim(Protocol):
    """Callable C-shim protocol for Wong-Wang ctypes facades."""

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


class FakeWongWangLib:
    """Shared-library stub exposing the Wong-Wang C symbol."""

    def __init__(self, return_code: int = 0) -> None:
        """Initialise the fake library with a C-shim return code."""
        self.wong_wang_simulate_c: WongWangCShim = FakeCShim(return_code)


class FakeJuliaWongWangModule:
    """Julia module stub exposing the Wong-Wang batch function."""

    def simulate_wong_wang_b(
        self,
        s1_init: float,
        s2_init: float,
        tau_s: float,
        gamma: float,
        j_n: float,
        j_cross: float,
        i_0: float,
        sigma: float,
        dt: float,
        stim1: npt.NDArray[np.float64],
        stim2: npt.NDArray[np.float64],
        xi: npt.NDArray[np.float64],
        s1_out: npt.NDArray[np.float64],
        s2_out: npt.NDArray[np.float64],
        r1_out: npt.NDArray[np.float64],
        r2_out: npt.NDArray[np.float64],
    ) -> tuple[float, float]:
        """Populate output buffers with a deterministic fake trajectory."""
        del tau_s, gamma, j_n, j_cross, i_0, sigma, dt, xi
        s1_out[:] = s1_init + stim1
        s2_out[:] = s2_init + stim2
        r1_out[:] = stim1
        r2_out[:] = stim2
        return float(s1_out[-1]) if s1_out.size else s1_init, (
            float(s2_out[-1]) if s2_out.size else s2_init
        )


def _call_dispatcher(
    dispatcher: WongDispatcher,
    stim1: npt.ArrayLike,
    stim2: npt.ArrayLike,
    xi: npt.ArrayLike,
) -> WongResult:
    """Invoke a Wong-Wang acceleration dispatcher with canonical parameters."""
    return dispatcher(
        0.1,
        0.2,
        0.1,
        0.641,
        0.2609,
        0.0497,
        0.3255,
        0.02,
        0.001,
        stim1,
        stim2,
        xi,
    )


@pytest.mark.parametrize(
    "module,dispatcher",
    [
        (go_wong, go_wong.simulate_wong_wang),
        (mojo_wong, mojo_wong.simulate_wong_wang),
    ],
)
@pytest.mark.parametrize(
    "stim1,stim2,xi,field",
    [
        (
            np.zeros((2, 2), dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            np.zeros(8, dtype=np.float64),
            "stim1",
        ),
        (
            np.zeros(4, dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
            np.zeros(8, dtype=np.float64),
            "stim2",
        ),
        (
            np.zeros(4, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            np.zeros((4, 2), dtype=np.float64),
            "xi",
        ),
    ],
)
def test_ctypes_wong_wang_rejects_non_1d_inputs(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    dispatcher: WongDispatcher,
    stim1: npt.NDArray[np.float64],
    stim2: npt.NDArray[np.float64],
    xi: npt.NDArray[np.float64],
    field: str,
) -> None:
    """Go and Mojo ctypes facades reject matrix-valued input vectors."""
    monkeypatch.setattr(module, "_lib", FakeWongWangLib())

    with pytest.raises(ValueError, match=f"{field} must be one-dimensional"):
        _call_dispatcher(dispatcher, stim1, stim2, xi)


@pytest.mark.parametrize(
    "stim1,stim2,xi,field",
    [
        (
            np.zeros((2, 2), dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            np.zeros(8, dtype=np.float64),
            "stim1",
        ),
        (
            np.zeros(4, dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
            np.zeros(8, dtype=np.float64),
            "stim2",
        ),
        (
            np.zeros(4, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            np.zeros((4, 2), dtype=np.float64),
            "xi",
        ),
    ],
)
def test_julia_wong_wang_rejects_non_1d_inputs(
    monkeypatch: pytest.MonkeyPatch,
    stim1: npt.NDArray[np.float64],
    stim2: npt.NDArray[np.float64],
    xi: npt.NDArray[np.float64],
    field: str,
) -> None:
    """Julia facade rejects matrix-valued inputs before backend calls."""
    monkeypatch.setattr(
        julia_neurons,
        "_ensure_wong_wang_loaded",
        lambda: FakeJuliaWongWangModule(),
    )

    with pytest.raises(ValueError, match=f"{field} must be one-dimensional"):
        _call_dispatcher(julia_neurons.simulate_wong_wang, stim1, stim2, xi)


@pytest.mark.parametrize(
    "module,dispatcher",
    [
        (go_wong, go_wong.simulate_wong_wang),
        (mojo_wong, mojo_wong.simulate_wong_wang),
    ],
)
def test_ctypes_wong_wang_stimulus_length_mismatch_raises(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    dispatcher: WongDispatcher,
) -> None:
    """Go and Mojo facades reject mismatched pool stimulus lengths."""
    monkeypatch.setattr(module, "_lib", FakeWongWangLib())

    with pytest.raises(ValueError, match="stim1 and stim2 length mismatch: 3 vs 2"):
        _call_dispatcher(
            dispatcher,
            np.zeros(3, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
        )


@pytest.mark.parametrize(
    "module,dispatcher",
    [
        (go_wong, go_wong.simulate_wong_wang),
        (mojo_wong, mojo_wong.simulate_wong_wang),
    ],
)
def test_ctypes_wong_wang_noise_length_mismatch_raises(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    dispatcher: WongDispatcher,
) -> None:
    """Go and Mojo facades reject noise traces without one pair per step."""
    monkeypatch.setattr(module, "_lib", FakeWongWangLib())

    with pytest.raises(ValueError, match="xi length must be 2 \\* n_steps \\(6\\): got 5"):
        _call_dispatcher(
            dispatcher,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.zeros(5, dtype=np.float64),
        )


@pytest.mark.parametrize(
    "module,dispatcher,error_message",
    [
        (go_wong, go_wong.simulate_wong_wang, "wong_wang_simulate_c"),
        (mojo_wong, mojo_wong.simulate_wong_wang, "Mojo wong_wang"),
    ],
)
def test_ctypes_wong_wang_nonzero_return_code_raises(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    dispatcher: WongDispatcher,
    error_message: str,
) -> None:
    """Go and Mojo facades report non-zero C ABI return codes."""
    monkeypatch.setattr(module, "_lib", FakeWongWangLib(return_code=42))

    with pytest.raises(RuntimeError, match=f"{error_message}.*42"):
        _call_dispatcher(
            dispatcher,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
        )


@pytest.mark.parametrize(
    "module,dispatcher",
    [
        (go_wong, go_wong.simulate_wong_wang),
        (mojo_wong, mojo_wong.simulate_wong_wang),
    ],
)
def test_ctypes_wong_wang_success_returns_one_dimensional_buffers(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    dispatcher: WongDispatcher,
) -> None:
    """Go and Mojo facades return one-dimensional output buffers on success."""
    monkeypatch.setattr(module, "_lib", FakeWongWangLib(return_code=0))

    out = _call_dispatcher(
        dispatcher,
        np.array([0.0, 1.0, 2.0], dtype=np.float64),
        np.array([2.0, 1.0, 0.0], dtype=np.float64),
        np.zeros(6, dtype=np.float64),
    )

    for key in ("s1", "s2", "r1", "r2"):
        assert isinstance(out[key], np.ndarray)
        assert out[key].shape == (3,)
        assert out[key].dtype == np.dtype(np.float64)
    assert out["s1_final"] == 0.0
    assert out["s2_final"] == 0.0


@pytest.mark.parametrize(
    "module,dispatcher,library_name",
    [
        (go_wong, go_wong.simulate_wong_wang, "libwong_wang.so not built"),
        (mojo_wong, mojo_wong.simulate_wong_wang, "libwong_wang.so not built"),
    ],
)
def test_ctypes_wong_wang_unavailable_library_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    dispatcher: WongDispatcher,
    library_name: str,
) -> None:
    """Go and Mojo facades raise an install hint when no shared library exists."""
    monkeypatch.setattr(module, "_lib", None)

    with pytest.raises(ImportError, match=library_name):
        _call_dispatcher(
            dispatcher,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
        )


@pytest.mark.parametrize("module", [go_wong, mojo_wong])
def test_ctypes_wong_wang_loader_failure_sets_unavailable_sentinel(
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
