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
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import numpy.typing as npt

from sc_neurocore.solvers import RK4Solver, RosenbrockEuler

# ───────────────────────── backend detection ─────────────────────────
#
# ``step`` advances one integrator update. ``simulate`` is an N-step sequential
# recurrence. The Rust engine class ``AdExNeuron`` implements the same
# factory-default baseline-Euler update as the pure-NumPy path and reproduces
# its voltage trace bit-for-bit under that contract (no parameter injection on
# the engine class). RK4 and Rosenbrock remain Python-only here.

_EngineAdEx = Any


def _load_engine_adex() -> type[Any]:
    engine = _importlib.import_module("sc_neurocore_engine")
    return engine.AdExNeuron  # type: ignore[no-any-return]


try:
    _EngineAdExCls: Optional[type[Any]] = _load_engine_adex()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _EngineAdExCls = None
    _HAS_RUST = False

# Factory defaults for the Rust engine path (no parameter/state injection).
_RUST_ENGINE_DEFAULTS: dict[str, float] = {
    "v": -65.0,
    "w": 0.0,
    "v_rest": -65.0,
    "v_reset": -68.0,
    "v_threshold": -50.0,
    "v_rh": -55.0,
    "delta_t": 2.0,
    "tau": 20.0,
    "tau_w": 100.0,
    "a": 0.5,
    "b": 7.0,
    "c_m": 200.0,
    "dt": 0.1,
}


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

    ``simulate`` supports ``backend`` values ``python``, ``rust``, and ``auto``
    (prefer Rust when the factory-default Euler contract holds and the engine
    wheel is present).
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

    def _matches_rust_engine_contract(self) -> bool:
        """Return whether the instance matches the Rust engine default contract.

        The engine ``AdExNeuron`` class has no parameter or state injection; it
        only reproduces the factory-default baseline-Euler trajectory.
        """
        if self.integrator != "baseline_euler":
            return False
        for name, expected in _RUST_ENGINE_DEFAULTS.items():
            if float(getattr(self, name)) != expected:
                return False
        return True

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
            ``"python"`` always uses the local ``step`` loop. ``"rust"`` uses
            the engine ``AdExNeuron`` under the factory-default baseline-Euler
            contract and raises when that contract is not met or the engine is
            unavailable. ``"auto"`` prefers Rust when the contract holds and the
            engine is present, otherwise Python.

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
            If ``backend="rust"`` is requested but the engine is missing or the
            instance is outside the default Euler contract.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if backend not in ("auto", "python", "rust"):
            raise ValueError(f"backend must be auto/python/rust, got {backend!r}")
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()

        prefer_rust = backend == "rust" or (
            backend == "auto" and _HAS_RUST and self._matches_rust_engine_contract()
        )
        if prefer_rust:
            if not _HAS_RUST or _EngineAdExCls is None:
                raise RuntimeError(
                    "Rust AdEx backend requested but sc_neurocore_engine is unavailable."
                )
            if not self._matches_rust_engine_contract():
                raise RuntimeError(
                    "Rust AdEx engine backend requires factory-default parameters, "
                    "baseline_euler integrator, and factory-default initial state "
                    f"(v={_RUST_ENGINE_DEFAULTS['v']}, w={_RUST_ENGINE_DEFAULTS['w']})."
                )
            trace, spikes, state = self._simulate_rust(n_steps, current)
        else:
            if backend == "rust":
                raise RuntimeError(
                    "Rust AdEx backend requested but sc_neurocore_engine is unavailable."
                )
            trace, spikes, state = self._simulate_python(n_steps, current)
        self.v, self.w = state
        return trace, spikes

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float]]:
        """Run the pure-Python step loop and return ``(trace, spikes, (v, w))``."""
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for t in range(n_steps):
            spikes += self.step(current)
            trace[t] = self.v
        return trace, spikes, (self.v, self.w)

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float]]:
        """Run the engine AdEx step loop under the default Euler contract."""
        assert _EngineAdExCls is not None
        neuron = _EngineAdExCls()
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for t in range(n_steps):
            spikes += int(neuron.step(float(current)))
            state = neuron.get_state()
            trace[t] = float(state["v"])
        final = neuron.get_state()
        return trace, spikes, (float(final["v"]), float(final["w"]))

    def reset(self) -> None:
        self.v = self.v_rest
        self.w = 0.0
