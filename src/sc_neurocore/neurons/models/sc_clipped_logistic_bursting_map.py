# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained clipped-logistic fast/slow bursting recurrence

"""Count-neutral retained two-state clipped-logistic bursting recurrence.

This project recurrence preserves the historical fast/slow map without a
whole-model publication attribution. Each step uses only finite arithmetic,
and the accelerated simulation paths preserve the historical trajectory.

    x(n+1) = clip(a*x(n)*(1 - x(n)) - y(n) + I, -2, 2)
    y(n+1) = y(n) + epsilon * (x(n) - sigma)

The source-faithful Cazelles-Courbage-Rabinovich identity lives in
``cazelles_map.py``.
"""

from __future__ import annotations

import ctypes
import importlib as _importlib
import math
import os as _os
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

# ───────────────────────── backend detection ─────────────────────────
#
# The single `step` is trivial, but an N-step simulation is a sequential
# recurrence (each step depends on the previous) that does not vectorise,
# so a compiled inner loop genuinely beats Python. The polyglot chain
# (Rust PyO3, Julia juliacall, Go cgo, Mojo FFI) accelerates `simulate`
# and reproduces the NumPy reference to the last bit.

_MAX_C_STEPS = (1 << 31) - 1


class _RustSimulate(Protocol):
    def __call__(self, *args: Any) -> tuple[list[float], int, float, float]: ...


class _JuliaResult(Protocol):
    trace: Any
    spikes: int
    xf: float
    yf: float


class _JuliaAccel(Protocol):
    def simulate_trace(self, *args: Any) -> _JuliaResult: ...


def _load_rust_simulate() -> _RustSimulate:
    engine = _importlib.import_module("sc_neurocore_engine")
    return cast(_RustSimulate, engine.py_sc_clipped_logistic_bursting_map_simulate)


try:
    _rust_simulate: _RustSimulate | None = _load_rust_simulate()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _rust_simulate = None
    _HAS_RUST = False

_julia_module: _JuliaAccel | None = None
_HAS_JULIA = False
_go_lib: ctypes.CDLL | None = None
_HAS_GO = False
_mojo_lib: ctypes.CDLL | None = None
_HAS_MOJO = False

_ACCEL_ROOT = _os.path.join(_os.path.dirname(__file__), "..", "..", "accel")


def _require_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_step_count(n_steps: object) -> int:
    if isinstance(n_steps, bool) or not isinstance(n_steps, Integral):
        raise TypeError("n_steps must be an integer")
    steps = int(n_steps)
    if steps < 0:
        raise ValueError("n_steps must be non-negative")
    if steps > _MAX_C_STEPS:
        raise ValueError(f"n_steps must be at most {_MAX_C_STEPS}")
    return steps


def _ensure_julia_loaded() -> bool:
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    import importlib.util as importlib_util

    if importlib_util.find_spec("juliacall") is None:
        return False
    jl_path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "julia", "neurons", "sc_clipped_logistic_bursting_map.jl")
    )
    if not _os.path.isfile(jl_path):
        return False
    juliacall = _importlib.import_module("juliacall")
    jl = juliacall.Main
    jl.include(jl_path)
    _julia_module = cast(_JuliaAccel, jl.SCClippedLogisticBurstingMapAccel)
    _HAS_JULIA = True
    return True


def _ensure_go_loaded() -> bool:
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True

    so_path = _os.path.abspath(
        _os.path.join(
            _ACCEL_ROOT,
            "go",
            "neurons",
            "sc_clipped_logistic_bursting_map",
            "libsc_clipped_logistic_bursting_map.so",
        )
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "sc_clipped_logistic_bursting_map_simulate_c", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.c_double] * 6 + [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    fn.restype = ctypes.c_longlong
    _go_lib = lib
    _HAS_GO = True
    return True


def _ensure_mojo_loaded() -> bool:
    from sc_neurocore.accel.mojo.native_library import load_mojo_library

    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True

    so_path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libsc_clipped_logistic_bursting_map.so")
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = load_mojo_library(so_path)
    except OSError:
        return False
    fn = getattr(lib, "sc_clipped_logistic_bursting_map_simulate_c", None)
    if fn is None:
        return False
    # 6 float params + n_steps + current addr + trace addr; returns spikes
    fn.argtypes = [ctypes.c_double] * 6 + [ctypes.c_int64, ctypes.c_double, ctypes.c_int64]
    fn.restype = ctypes.c_int64
    _mojo_lib = lib
    _HAS_MOJO = True
    return True


@dataclass
class SCClippedLogisticBurstingMapNeuron:
    """Retained project-defined two-state clipped-logistic bursting recurrence.

    x(n+1) = f(x(n)) - y(n) + I
    y(n+1) = y(n) + epsilon * (x(n) - sigma)

    f(x) = a*x*(1 - x)    (logistic-like fast dynamics)

    Bursting arises from slow y modulation of fast x.

    This is a project-defined recurrence without whole-model publication
    attribution.
    """

    x: float = 0.1
    y: float = 0.0
    a: float = 3.8
    epsilon: float = 0.01
    sigma: float = 0.5
    x_threshold: float = 0.9

    def __post_init__(self) -> None:
        """Validate initial state and parameters.

        Raises
        ------
        TypeError
            If any state or parameter is a boolean or non-real value.
        ValueError
            If any state or parameter is NaN or infinite.
        """
        self.x = _require_finite_float(self.x, "x")
        self.y = _require_finite_float(self.y, "y")
        self.a = _require_finite_float(self.a, "a")
        self.epsilon = _require_finite_float(self.epsilon, "epsilon")
        self.sigma = _require_finite_float(self.sigma, "sigma")
        self.x_threshold = _require_finite_float(self.x_threshold, "x_threshold")

    def _validated_state_and_params(self) -> tuple[float, float, float, float, float, float]:
        x = self.x
        y = self.y
        a = self.a
        epsilon = self.epsilon
        sigma = self.sigma
        x_threshold = self.x_threshold
        if (
            type(x) is float
            and type(y) is float
            and type(a) is float
            and type(epsilon) is float
            and type(sigma) is float
            and type(x_threshold) is float
            and x - x == 0.0
            and y - y == 0.0
            and a - a == 0.0
            and epsilon - epsilon == 0.0
            and sigma - sigma == 0.0
            and x_threshold - x_threshold == 0.0
        ):
            return x, y, a, epsilon, sigma, x_threshold
        return (
            _require_finite_float(x, "x"),
            _require_finite_float(y, "y"),
            _require_finite_float(a, "a"),
            _require_finite_float(epsilon, "epsilon"),
            _require_finite_float(sigma, "sigma"),
            _require_finite_float(x_threshold, "x_threshold"),
        )

    def step(self, current: float) -> int:
        """Advance the clipped logistic map one step.

        Parameters
        ----------
        current : float
            Finite external input drive.

        Returns
        -------
        int
            ``1`` when the updated ``x`` is at or above ``x_threshold``,
            else ``0``.

        Raises
        ------
        TypeError
            If ``current`` or any model state or parameter is a boolean or
            non-real value.
        ValueError
            If ``current``, any model state or parameter, or a candidate next
            state is NaN or infinite. Rejected updates leave ``(x, y)``
            unchanged.
        """
        if type(current) is float and current - current == 0.0:
            drive = current
        else:
            drive = _require_finite_float(current, "current")
        x, y, a, epsilon, sigma, x_threshold = self._validated_state_and_params()
        f = a * x * (1.0 - x)
        x_new = f - y + drive
        y_new = y + epsilon * (x - sigma)
        if not (x_new - x_new == 0.0 and y_new - y_new == 0.0):
            raise ValueError(
                "SCClippedLogisticBurstingMapNeuron update overflowed to non-finite state"
            )
        x_next = min(2.0, max(-2.0, x_new))
        self.x = x_next
        self.y = y_new
        return 1 if x_next >= x_threshold else 0

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` from the current state, returning ``(trace, spikes)``.

        ``trace[t]`` is the membrane variable ``x`` after step ``t``; ``spikes``
        counts the steps whose ``x`` reached ``x_threshold``. The instance state
        ``(x, y)`` is advanced to the final step only after finite results are
        verified. Rust, Julia, and Go reproduce the Python reference bit-for-bit;
        Mojo agrees within ULP bounds.

        Parameters
        ----------
        n_steps : int
            Number of simulation steps to run (``0 <= n_steps <= 2**31 - 1``).
        current : float, optional
            Finite constant external input drive (default ``0.0``).
        backend : str, optional
            Execution backend: ``"auto"``, ``"rust"``, ``"julia"``, ``"go"``,
            ``"mojo"``, or ``"python"`` (default ``"auto"``).

        Returns
        -------
        tuple[numpy.typing.NDArray[numpy.float64], int]
            ``(trace, spikes)`` pair for the simulated horizon.

        Raises
        ------
        TypeError
            If ``n_steps`` is a boolean or non-integer, or if ``current`` or
            any state or parameter is a boolean or non-real value.
        ValueError
            If ``n_steps`` is negative or exceeds ``2**31 - 1``, ``backend`` is
            unknown, ``current`` or any state or parameter is non-finite, or the
            trajectory overflows to a non-finite state. Rejected calls leave
            ``(x, y)`` unchanged.
        RuntimeError
            If an explicitly requested compiled backend is unavailable.
        """
        steps = _require_step_count(n_steps)
        drive = _require_finite_float(current, "current")
        if not isinstance(backend, str) or backend not in (
            "auto",
            "rust",
            "julia",
            "go",
            "mojo",
            "python",
        ):
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")
        self._validated_state_and_params()

        if backend == "rust" and not _HAS_RUST:
            raise RuntimeError(
                "Rust SC clipped-logistic backend requested but the engine wheel lacks it."
            )
        if backend == "julia" and not _ensure_julia_loaded():
            raise RuntimeError(
                "Julia SC clipped-logistic backend requested but juliacall/.jl is unavailable."
            )
        if backend == "go" and not _ensure_go_loaded():
            raise RuntimeError(
                "Go SC clipped-logistic backend requested but libsc_clipped_logistic_bursting_map.so is not built; run "
                "`cd src/sc_neurocore/accel/go/neurons/sc_clipped_logistic_bursting_map && go build "
                "-buildmode=c-shared -o libsc_clipped_logistic_bursting_map.so sc_clipped_logistic_bursting_map.go`."
            )
        if backend == "mojo" and not _ensure_mojo_loaded():
            raise RuntimeError(
                "Mojo SC clipped-logistic backend requested but libsc_clipped_logistic_bursting_map.so is not built; run "
                "`cd src/sc_neurocore/accel/mojo/neurons && mojo build --emit shared-lib "
                "-o libsc_clipped_logistic_bursting_map.so sc_clipped_logistic_bursting_map.mojo`."
            )

        if backend == "rust" or (backend == "auto" and _HAS_RUST):
            trace, spikes, xf, yf = self._simulate_rust(steps, drive)
        elif backend == "julia":
            trace, spikes, xf, yf = self._simulate_julia(steps, drive)
        elif backend == "go":
            trace, spikes, xf, yf = self._simulate_go(steps, drive)
        elif backend == "mojo":
            trace, spikes, xf, yf = self._simulate_mojo(steps, drive)
        else:
            trace, spikes, xf, yf = self._simulate_python(steps, drive)

        if not (
            math.isfinite(xf)
            and math.isfinite(yf)
            and (steps == 0 or math.isfinite(float(trace.sum())))
        ):
            raise ValueError(
                "SCClippedLogisticBurstingMapNeuron simulation overflowed to non-finite state"
            )
        self.x, self.y = xf, yf
        return trace, spikes

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        x, y, a, epsilon, sigma, x_threshold = self._validated_state_and_params()
        drive = _require_finite_float(current, "current")
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for t in range(n_steps):
            f = a * x * (1.0 - x)
            x_new = f - y + drive
            y_new = y + epsilon * (x - sigma)
            if not (x_new - x_new == 0.0 and y_new - y_new == 0.0):
                raise ValueError(
                    "SCClippedLogisticBurstingMapNeuron update overflowed to non-finite state"
                )
            x = min(2.0, max(-2.0, x_new))
            y = y_new
            trace[t] = x
            if x >= x_threshold:
                spikes += 1
        return trace, spikes, x, y

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _rust_simulate is not None
        x, y, a, epsilon, sigma, x_threshold = self._validated_state_and_params()
        trace_list, spikes, xf, yf = _rust_simulate(
            x, y, a, epsilon, sigma, x_threshold, n_steps, current
        )
        return np.asarray(trace_list, dtype=np.float64), int(spikes), float(xf), float(yf)

    def _simulate_julia(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _julia_module is not None
        x, y, a, epsilon, sigma, x_threshold = self._validated_state_and_params()
        result = _julia_module.simulate_trace(
            x,
            y,
            a,
            epsilon,
            sigma,
            x_threshold,
            int(n_steps),
            float(current),
        )
        trace = np.asarray(result.trace, dtype=np.float64)
        return trace, int(result.spikes), float(result.xf), float(result.yf)

    def _simulate_go(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _go_lib is not None
        x, y, a, epsilon, sigma, x_threshold = self._validated_state_and_params()
        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        spikes = _go_lib.sc_clipped_logistic_bursting_map_simulate_c(
            ctypes.c_double(x),
            ctypes.c_double(y),
            ctypes.c_double(a),
            ctypes.c_double(epsilon),
            ctypes.c_double(sigma),
            ctypes.c_double(x_threshold),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        xf = float(trace[n_steps]) if n_steps > 0 else x
        yf = float(trace[n_steps + 1])
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), xf, yf

    def _simulate_mojo(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _mojo_lib is not None
        x, y, a, epsilon, sigma, x_threshold = self._validated_state_and_params()
        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        spikes = _mojo_lib.sc_clipped_logistic_bursting_map_simulate_c(
            x,
            y,
            a,
            epsilon,
            sigma,
            x_threshold,
            int(n_steps),
            float(current),
            int(trace.ctypes.data),
        )
        xf = float(trace[n_steps]) if n_steps > 0 else x
        yf = float(trace[n_steps + 1])
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), xf, yf

    def reset(self) -> None:
        """Restore the default state ``x = 0.1``, ``y = 0.0``."""
        self.x = 0.1
        self.y = 0.0
