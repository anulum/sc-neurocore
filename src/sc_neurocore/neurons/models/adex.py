# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive Exponential Integrate-and-Fire. Brette &

from __future__ import annotations

import importlib as _importlib
import math
import os as _os
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from sc_neurocore.solvers import RK4Solver, RosenbrockEuler

# ───────────────────────── backend detection ─────────────────────────
#
# ``step`` advances one integrator update. ``simulate`` is an N-step sequential
# recurrence and therefore benefits from a compiled inner loop. The Rust engine
# binding and Julia, Go, and Mojo accept the complete maintained numeric state
# and parameter surface through failure-atomic batches. RK4
# and Rosenbrock remain on their already-established polyglot dispatcher and the
# local Python path; these model-specific kernels implement baseline Euler.


def _load_engine_adex() -> tuple[type[Any], Any]:
    engine = _importlib.import_module("sc_neurocore_engine")
    return engine.AdExNeuron, engine.adex_simulate_complete


_EngineAdExCls: type[Any] | None
_EngineAdExSimulateFn: Any | None
try:
    _EngineAdExCls, _EngineAdExSimulateFn = _load_engine_adex()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _EngineAdExCls = None
    _EngineAdExSimulateFn = None
    _HAS_RUST = False

_julia_module: Any | None = None
_HAS_JULIA = False
_go_lib: Any | None = None
_HAS_GO = False
_mojo_lib: Any | None = None
_HAS_MOJO = False

_ACCEL_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "accel"))


def _ensure_julia_loaded() -> bool:
    """Load the executable AdEx Julia module when its runtime is available."""
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    import importlib.util as importlib_util

    if importlib_util.find_spec("juliacall") is None:
        return False
    source_path = _os.path.join(_ACCEL_ROOT, "julia", "neurons", "adex.jl")
    if not _os.path.isfile(source_path):
        return False
    try:
        juliacall = _importlib.import_module("juliacall")
        julia = juliacall.Main
        julia.include(source_path)
        _julia_module = julia.AdexAccel
    except (ImportError, AttributeError, RuntimeError):
        return False
    _HAS_JULIA = True
    return True


def _ensure_go_loaded() -> bool:
    """Load the compiled AdEx Go C-ABI bridge when it is available."""
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    import ctypes

    library_path = _os.path.join(_ACCEL_ROOT, "go", "neurons", "adex", "libadex.so")
    if not _os.path.isfile(library_path):
        return False
    try:
        library = ctypes.CDLL(library_path)
    except OSError:
        return False
    simulate = getattr(library, "adex_simulate_complete_c", None)
    if simulate is None:
        return False
    simulate.argtypes = [ctypes.c_double] * 13 + [
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    simulate.restype = ctypes.c_int64
    compatibility = getattr(library, "adex_simulate_c", None)
    if compatibility is not None:
        compatibility.argtypes = [ctypes.c_double] * 13 + [
            ctypes.c_int64,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
        ]
        compatibility.restype = ctypes.c_int64
    _go_lib = library
    _HAS_GO = True
    return True


def _ensure_mojo_loaded() -> bool:
    """Load the compiled AdEx Mojo C ABI when it is available."""
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    import ctypes

    library_path = _os.path.join(_ACCEL_ROOT, "mojo", "kernels", "libadex.so")
    if not _os.path.isfile(library_path):
        return False
    try:
        library = ctypes.CDLL(library_path)
    except OSError:
        return False
    simulate = getattr(library, "adex_simulate_complete_c", None)
    if simulate is None:
        return False
    simulate.argtypes = [ctypes.c_double] * 13 + [
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    simulate.restype = ctypes.c_int64
    compatibility = getattr(library, "adex_simulate_c", None)
    if compatibility is not None:
        compatibility.argtypes = [ctypes.c_double] * 13 + [
            ctypes.c_int64,
            ctypes.c_double,
            ctypes.c_int64,
        ]
        compatibility.restype = ctypes.c_int64
    _mojo_lib = library
    _HAS_MOJO = True
    return True


@dataclass
class AdExNeuron:
    """Adaptive Exponential Integrate-and-Fire. Brette & Gerstner 2005.

    dv/dt = -(v - v_rest)/tau + delta_T * exp((v - v_rh)/delta_T) / tau - w/C + I/C
    dw/dt = (a * (v - v_rest) - w) / tau_w
    if v >= v_threshold: v = v_reset, w += b

    Reference: Brette, R. & Gerstner, W. (2005). J. Neurophysiol. 94:3637–3642.

    Integrator options:
    - ``baseline_euler`` preserves the historical explicit-Euler path
    - ``rk4`` is an explicit higher-order alternative path
    - ``rosenbrock`` is a linearly implicit stiff-system path over the same
      AdEx ODEs

    ``simulate`` exposes the baseline-Euler Python, Rust, Julia, Go and Mojo
    paths. ``auto`` follows the committed measured order Rust, Julia, Go,
    Mojo, then Python.
    """

    v: float = -65.0
    w: float = 0.0
    v_rest: float = -65.0
    v_reset: float = -68.0
    v_threshold: float = -50.0
    v_rh: float = -55.0
    delta_t: float = 2.0
    tau: float = 20.0
    tau_w: float = 100.0
    a: float = 0.5
    b: float = 7.0
    c_m: float = 200.0
    dt: float = 0.1
    integrator: Literal["baseline_euler", "rk4", "rosenbrock"] = "baseline_euler"

    def __post_init__(self) -> None:
        if self.integrator not in {"baseline_euler", "rk4", "rosenbrock"}:
            raise ValueError(f"Unsupported integrator for AdExNeuron: {self.integrator}")
        for field in ("v", "w", "v_rest", "v_reset", "v_threshold", "v_rh", "a", "b"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        for field in ("delta_t", "tau", "tau_w", "c_m", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")

    @classmethod
    def brette_gerstner_2005(
        cls,
        *,
        dt: float = 0.1,
        integrator: Literal["baseline_euler", "rk4", "rosenbrock"] = "baseline_euler",
    ) -> AdExNeuron:
        """Return the regular-spiking fit reported by Brette and Gerstner (2005).

        The paper reports ``C=281 pF``, ``gL=30 nS``, ``EL=-70.6 mV``,
        ``VT=-50.4 mV``, ``DeltaT=2 mV``, ``tau_w=144 ms``, ``a=4 nS``,
        ``b=80.5 pA``, numerical spike cutoff ``Vpeak=20 mV``, and reset
        ``Vr=EL``.  This implementation stores ``tau=C/gL``.  ``dt`` and the
        integrator are explicit numerical choices because the paper defines a
        continuous-time model, not a discrete timestep.
        """
        capacitance = 281.0
        leak_conductance = 30.0
        rest = -70.6
        return cls(
            v=rest,
            w=0.0,
            v_rest=rest,
            v_reset=rest,
            v_threshold=20.0,
            v_rh=-50.4,
            delta_t=2.0,
            tau=capacitance / leak_conductance,
            tau_w=144.0,
            a=4.0,
            b=80.5,
            c_m=capacitance,
            dt=dt,
            integrator=integrator,
        )

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()
        if self.integrator == "baseline_euler":
            next_v, next_w = self._step_baseline_euler(current)
        elif self.integrator == "rk4":
            next_v, next_w = self._step_rk4(current)
        else:
            next_v, next_w = self._step_rosenbrock(current)

        self._validate_update(next_v, next_w)
        if next_v >= self.v_threshold:
            spike_w = next_w + self.b
            if not math.isfinite(spike_w):
                raise ValueError("spike adaptation update must remain finite")
            self.v = self.v_reset
            self.w = spike_w
            return 1
        self.v = next_v
        self.w = next_w
        return 0

    def _validate_runtime_state(self) -> None:
        if not math.isfinite(self.v):
            raise ValueError("runtime voltage state must be finite")
        if not math.isfinite(self.w):
            raise ValueError("runtime adaptation state must be finite")

    def _validate_update(self, next_v: float, next_w: float) -> None:
        if not math.isfinite(next_v) or not math.isfinite(next_w):
            raise ValueError("AdEx integrator update must remain finite")

    def _rhs(self, _t: float, state: np.ndarray[Any, Any], current: float) -> np.ndarray[Any, Any]:
        v = float(state[0])
        w = float(state[1])
        exp_term = self.delta_t * np.exp(np.clip((v - self.v_rh) / self.delta_t, -20.0, 20.0))
        dv = (-(v - self.v_rest) + exp_term) / self.tau + (-w + current) / self.c_m
        dw = (self.a * (v - self.v_rest) - w) / self.tau_w
        return np.array([dv, dw], dtype=np.float64)

    def _step_baseline_euler(self, current: float) -> tuple[float, float]:
        with np.errstate(over="ignore", invalid="ignore"):
            exp_term = self.delta_t * np.exp(
                np.clip((self.v - self.v_rh) / self.delta_t, -20.0, 20.0)
            )
            dv = (-(self.v - self.v_rest) + exp_term) / self.tau + (-self.w + current) / self.c_m
            dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w
            return self.v + dv * self.dt, self.w + dw * self.dt

    def _step_rk4(self, current: float) -> tuple[float, float]:
        solver = RK4Solver()
        state = np.array([self.v, self.w], dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            state, _ = solver.step(
                lambda time, y: self._rhs(time, y, current),
                state,
                0.0,
                self.dt,
            )
        return float(state[0]), float(state[1])

    def _step_rosenbrock(self, current: float) -> tuple[float, float]:
        solver = RosenbrockEuler()
        state = np.array([self.v, self.w], dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            state, _ = solver.step(
                lambda time, y: self._rhs(time, y, current),
                state,
                0.0,
                self.dt,
            )
        return float(state[0]), float(state[1])

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` updates from the current state, returning ``(trace, spikes)``.

        ``trace[t]`` is the membrane voltage ``v`` after step ``t`` (post-reset
        when that step fired); ``spikes`` counts the threshold crossings. The
        instance state ``(v, w)`` is advanced to the final step.

        Parameters
        ----------
        n_steps:
            Number of integrator updates (must be non-negative).
        current:
            Constant injected current for every step (must be finite).
        backend:
            ``"python"`` always uses the local ``step`` loop. ``"rust"``,
            ``"julia"``, ``"go"`` and ``"mojo"`` run their checked compiled
            baseline-Euler batches with the complete maintained numeric state
            and parameter surface. ``"auto"`` chooses an available compiled
            baseline-Euler path before the Python floor.

        Returns
        -------
        tuple[npt.NDArray[np.float64], int]
            Voltage trace of length ``n_steps`` and total spike count.

        Raises
        ------
        ValueError
            If ``n_steps`` is negative, ``current`` is non-finite, or ``backend``
            is unknown.
        RuntimeError
            If a requested compiled backend is unavailable or the selected
            backend does not support the configured integrator contract.
        """
        v_trace, _w_trace, event_trace = self.simulate_complete(n_steps, current, backend)
        return v_trace, int(np.sum(event_trace, dtype=np.int64))

    def simulate_complete(
        self,
        n_steps: int,
        current: float = 0.0,
        backend: str = "auto",
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.uint8],
    ]:
        """Return aligned post-step voltage, adaptation, and event traces.

        The receiver is updated only after the complete selected-backend batch
        has succeeded and its packet has passed shape, finiteness, event-domain,
        and final-state validation.  A rejected later step therefore cannot
        leave a partially advanced neuron.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if backend not in ("auto", "python", "rust", "julia", "go", "mojo"):
            raise ValueError(f"backend must be auto/python/rust/julia/go/mojo, got {backend!r}")
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()

        if backend in {"rust", "julia", "go", "mojo"} and self.integrator != "baseline_euler":
            raise RuntimeError(
                f"{backend.title()} AdEx backend requires baseline_euler integrator."
            )

        selected = backend
        if backend == "auto":
            if self.integrator != "baseline_euler":
                selected = "python"
            elif _HAS_RUST:
                selected = "rust"
            elif _ensure_julia_loaded():
                selected = "julia"
            elif _ensure_go_loaded():
                selected = "go"
            elif _ensure_mojo_loaded():
                selected = "mojo"
            else:
                selected = "python"

        if selected == "rust":
            if not _HAS_RUST or _EngineAdExSimulateFn is None:
                raise RuntimeError(
                    "Rust AdEx backend requested but sc_neurocore_engine is unavailable."
                )
            packet = self._simulate_rust_complete(n_steps, current)
        elif selected == "julia":
            if not _ensure_julia_loaded():
                raise RuntimeError(
                    "Julia AdEx backend requested but juliacall or the AdEx module is unavailable."
                )
            packet = self._simulate_julia_complete(n_steps, current)
        elif selected == "go":
            if not _ensure_go_loaded():
                raise RuntimeError(
                    "Go AdEx backend requested but libadex.so is not built; run "
                    "go build -buildmode=c-shared -o libadex.so adex.go in "
                    "accel/go/neurons/adex."
                )
            packet = self._simulate_go_complete(n_steps, current)
        elif selected == "mojo":
            if not _ensure_mojo_loaded():
                raise RuntimeError(
                    "Mojo AdEx backend requested but libadex.so is not built; run "
                    "mojo build --emit shared-lib -o libadex.so adex.mojo in "
                    "accel/mojo/kernels."
                )
            packet = self._simulate_mojo_complete(n_steps, current)
        else:
            packet = self._simulate_python_complete(n_steps, current)

        v_trace, w_trace, event_trace, final_state = self._validated_complete_packet(
            packet, n_steps
        )
        self.v, self.w = final_state
        return v_trace, w_trace, event_trace

    @staticmethod
    def _validated_complete_packet(
        packet: tuple[object, object, object, tuple[float, float]],
        n_steps: int,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.uint8],
        tuple[float, float],
    ]:
        """Normalise one backend packet and reject malformed output atomically."""
        raw_v, raw_w, raw_events, final_state = packet
        v_trace = np.ascontiguousarray(np.asarray(raw_v, dtype=np.float64))
        w_trace = np.ascontiguousarray(np.asarray(raw_w, dtype=np.float64))
        event_values = np.asarray(raw_events)
        if v_trace.shape != (n_steps,) or w_trace.shape != (n_steps,):
            raise FloatingPointError("AdEx backend returned malformed state trace shapes.")
        if event_values.shape != (n_steps,):
            raise FloatingPointError("AdEx backend returned a malformed event trace shape.")
        if not np.all(np.isfinite(v_trace)) or not np.all(np.isfinite(w_trace)):
            raise FloatingPointError("AdEx backend returned non-finite trace state.")
        if not np.all((event_values == 0) | (event_values == 1)):
            raise FloatingPointError("AdEx backend returned events outside the binary domain.")
        event_trace = np.ascontiguousarray(event_values, dtype=np.uint8)
        final_v, final_w = map(float, final_state)
        if not math.isfinite(final_v) or not math.isfinite(final_w):
            raise FloatingPointError("AdEx backend returned a non-finite final state.")
        if n_steps and (final_v != float(v_trace[-1]) or final_w != float(w_trace[-1])):
            raise FloatingPointError("AdEx backend final state disagrees with its trace packet.")
        return v_trace, w_trace, event_trace, (final_v, final_w)

    def _simulate_python_complete(
        self, n_steps: int, current: float
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.uint8],
        tuple[float, float],
    ]:
        """Run a checked candidate copy so the Python batch is failure-atomic."""
        candidate = replace(self)
        v_trace = np.empty(n_steps, dtype=np.float64)
        w_trace = np.empty(n_steps, dtype=np.float64)
        events = np.empty(n_steps, dtype=np.uint8)
        for index in range(n_steps):
            events[index] = candidate.step(current)
            v_trace[index] = candidate.v
            w_trace[index] = candidate.w
        return v_trace, w_trace, events, (candidate.v, candidate.w)

    def _simulate_rust_complete(
        self, n_steps: int, current: float
    ) -> tuple[object, object, object, tuple[float, float]]:
        """Run the checked production-Rust full-parameter batch."""
        assert _EngineAdExSimulateFn is not None
        result = _EngineAdExSimulateFn(
            float(self.v),
            float(self.w),
            float(self.v_rest),
            float(self.v_reset),
            float(self.v_threshold),
            float(self.v_rh),
            float(self.delta_t),
            float(self.tau),
            float(self.tau_w),
            float(self.a),
            float(self.b),
            float(self.c_m),
            float(self.dt),
            int(n_steps),
            float(current),
        )
        return result[0], result[1], result[2], (float(result[3]), float(result[4]))

    def _simulate_julia_complete(
        self, n_steps: int, current: float
    ) -> tuple[object, object, object, tuple[float, float]]:
        """Run the Julia full-state baseline-Euler batch."""
        assert _julia_module is not None
        result = _julia_module.simulate_complete(
            float(self.v),
            float(self.w),
            float(self.v_rest),
            float(self.v_reset),
            float(self.v_threshold),
            float(self.v_rh),
            float(self.delta_t),
            float(self.tau),
            float(self.tau_w),
            float(self.a),
            float(self.b),
            float(self.c_m),
            float(self.dt),
            int(n_steps),
            float(current),
        )
        return result.v_trace, result.w_trace, result.events, (float(result.vf), float(result.wf))

    def _simulate_go_complete(
        self, n_steps: int, current: float
    ) -> tuple[object, object, object, tuple[float, float]]:
        """Run the checked Go complete-state C-ABI batch."""
        assert _go_lib is not None
        import ctypes

        v_trace = np.empty(n_steps + 1, dtype=np.float64)
        w_trace = np.empty(n_steps + 1, dtype=np.float64)
        events = np.empty(n_steps, dtype=np.uint8)
        spikes = int(
            _go_lib.adex_simulate_complete_c(
                self.v,
                self.w,
                self.v_rest,
                self.v_reset,
                self.v_threshold,
                self.v_rh,
                self.delta_t,
                self.tau,
                self.tau_w,
                self.a,
                self.b,
                self.c_m,
                self.dt,
                n_steps,
                current,
                v_trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                w_trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                events.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            )
        )
        if spikes < 0:
            raise FloatingPointError("Go AdEx kernel rejected the simulation contract.")
        if spikes != int(np.sum(events, dtype=np.int64)):
            raise FloatingPointError("Go AdEx event count disagrees with its event trace.")
        return (
            v_trace[:n_steps],
            w_trace[:n_steps],
            events,
            (
                float(v_trace[n_steps]),
                float(w_trace[n_steps]),
            ),
        )

    def _simulate_mojo_complete(
        self, n_steps: int, current: float
    ) -> tuple[object, object, object, tuple[float, float]]:
        """Run the failure-atomic Mojo complete-state C-ABI batch."""
        assert _mojo_lib is not None
        v_trace = np.empty(n_steps + 1, dtype=np.float64)
        w_trace = np.empty(n_steps + 1, dtype=np.float64)
        events = np.empty(n_steps, dtype=np.uint8)
        spikes = int(
            _mojo_lib.adex_simulate_complete_c(
                self.v,
                self.w,
                self.v_rest,
                self.v_reset,
                self.v_threshold,
                self.v_rh,
                self.delta_t,
                self.tau,
                self.tau_w,
                self.a,
                self.b,
                self.c_m,
                self.dt,
                n_steps,
                current,
                int(v_trace.ctypes.data),
                int(w_trace.ctypes.data),
                int(events.ctypes.data),
            )
        )
        if spikes < 0:
            raise FloatingPointError("Mojo AdEx kernel rejected the simulation contract.")
        if spikes != int(np.sum(events, dtype=np.int64)):
            raise FloatingPointError("Mojo AdEx event count disagrees with its event trace.")
        return (
            v_trace[:n_steps],
            w_trace[:n_steps],
            events,
            (
                float(v_trace[n_steps]),
                float(w_trace[n_steps]),
            ),
        )

    def reset(self) -> None:
        self.v = self.v_rest
        self.w = 0.0
