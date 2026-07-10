# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Connor-Stevens 1977 — A-type potassium current, Type-I

from __future__ import annotations

import importlib as _importlib
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import numpy as np
import numpy.typing as npt

try:
    _EngineCls: Optional[type[Any]] = _importlib.import_module(
        "sc_neurocore_engine"
    ).ConnorStevensNeuron
    _HAS_RUST = True
except (ImportError, AttributeError):
    _EngineCls = None
    _HAS_RUST = False

_RUST_ENGINE_DEFAULTS: dict[str, float] = {
    "v": -68.0,
    "m": 0.01,
    "h": 0.99,
    "n": 0.1,
    "a": 0.5,
    "b": 0.1,
    "g_na": 120.0,
    "g_k": 20.0,
    "g_a": 47.7,
    "g_l": 0.3,
    "e_na": 55.0,
    "e_k": -72.0,
    "e_a": -75.0,
    "e_l": -17.0,
    "c_m": 1.0,
    "dt": 0.01,
    "v_threshold": 0.0,
}


@dataclass
class ConnorStevensNeuron:
    """Connor-Stevens 1977 A-type potassium current model.

    State variables are membrane voltage ``v`` plus sodium activation ``m``,
    sodium inactivation ``h``, delayed-rectifier potassium activation ``n``,
    A-type potassium activation ``a``, and A-type inactivation ``b``. One
    public ``step`` advances the 1 ms macro-step with candidate-first RK4
    sub-steps and commits only finite, physically bounded candidates.
    """

    _EXP_MAX: ClassVar[float] = 700.0

    v: float = -68.0
    m: float = 0.01
    h: float = 0.99
    n: float = 0.1
    a: float = 0.5
    b: float = 0.1
    g_na: float = 120.0
    g_k: float = 20.0
    g_a: float = 47.7
    g_l: float = 0.3
    e_na: float = 55.0
    e_k: float = -72.0
    e_a: float = -75.0
    e_l: float = -17.0
    c_m: float = 1.0
    dt: float = 0.01
    v_threshold: float = 0.0

    def __post_init__(self) -> None:
        self.v = self._finite_float("v", self.v)
        self.m = self._gate_float("m", self.m, upper=1.0)
        self.h = self._gate_float("h", self.h, upper=1.0)
        self.n = self._gate_float("n", self.n, upper=1.0)
        self.a = self._gate_float("a", self.a, upper=1.5)
        self.b = self._gate_float("b", self.b, upper=1.0)
        self.g_na = self._non_negative_float("g_na", self.g_na)
        self.g_k = self._non_negative_float("g_k", self.g_k)
        self.g_a = self._non_negative_float("g_a", self.g_a)
        self.g_l = self._non_negative_float("g_l", self.g_l)
        self.e_na = self._finite_float("e_na", self.e_na)
        self.e_k = self._finite_float("e_k", self.e_k)
        self.e_a = self._finite_float("e_a", self.e_a)
        self.e_l = self._finite_float("e_l", self.e_l)
        self.c_m = self._positive_float("c_m", self.c_m)
        self.dt = self._positive_float("dt", self.dt)
        self.v_threshold = self._finite_float("v_threshold", self.v_threshold)

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a finite float")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    @classmethod
    def _positive_float(cls, name: str, value: float) -> float:
        value = cls._finite_float(name, value)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
        return value

    @classmethod
    def _non_negative_float(cls, name: str, value: float) -> float:
        value = cls._finite_float(name, value)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @classmethod
    def _gate_float(cls, name: str, value: float, *, upper: float) -> float:
        value = cls._finite_float(name, value)
        if not 0.0 <= value <= upper:
            raise ValueError(f"{name} must be within [0, {upper}]")
        return value

    @classmethod
    def _checked_exp(cls, x: float, name: str) -> float:
        if not math.isfinite(x):
            raise FloatingPointError(f"{name} exponent must be finite")
        if x > cls._EXP_MAX:
            raise FloatingPointError(f"{name} exponent overflow")
        return math.exp(x)

    @classmethod
    def _safe_rate(cls, scale: float, shift: float, v: float, denom: float, name: str) -> float:
        delta = v + shift
        x = delta / denom
        if abs(x) < 1e-9:
            return scale * denom
        value = scale * delta / (1.0 - cls._checked_exp(-x, name))
        if not math.isfinite(value):
            raise FloatingPointError(f"{name} rate is non-finite")
        return value

    @classmethod
    def _rates(
        cls, v: float
    ) -> tuple[float, float, float, float, float, float, float, float, float, float]:
        alpha_m = cls._safe_rate(0.38, 29.7, v, 10.0, "alpha_m")
        beta_m = 15.2 * cls._checked_exp(-(v + 54.7) / 18.0, "beta_m")
        alpha_h = 0.266 * cls._checked_exp(-(v + 48.0) / 20.0, "alpha_h")
        beta_h = 3.8 / (1.0 + cls._checked_exp(-(v + 18.0) / 10.0, "beta_h"))
        alpha_n = cls._safe_rate(0.02, 45.7, v, 10.0, "alpha_n")
        beta_n = 0.25 * cls._checked_exp(-(v + 55.7) / 80.0, "beta_n")
        a_inf_base = (
            0.0761
            * cls._checked_exp((v + 94.22) / 31.84, "a_inf_num")
            / (1.0 + cls._checked_exp((v + 1.17) / 28.93, "a_inf_den"))
        )
        if a_inf_base < 0.0 or not math.isfinite(a_inf_base):
            raise FloatingPointError("a_inf base is outside the real finite domain")
        a_inf = a_inf_base ** (1.0 / 3.0)
        tau_a = 0.3632 + 1.158 / (1.0 + cls._checked_exp((v + 55.96) / 20.12, "tau_a"))
        b_base = 1.0 / (1.0 + cls._checked_exp((v + 53.3) / 14.54, "b_inf"))
        b_inf = b_base**4
        tau_b = 1.24 + 2.678 / (1.0 + cls._checked_exp((v + 50.0) / 16.027, "tau_b"))
        rates = (alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n, a_inf, tau_a, b_inf, tau_b)
        if not all(math.isfinite(rate) for rate in rates) or tau_a <= 0.0 or tau_b <= 0.0:
            raise FloatingPointError(
                "Connor-Stevens rates must be finite with positive time constants"
            )
        return rates

    def _validate_runtime_state(self) -> None:
        self._finite_float("v", self.v)
        self._finite_float("m", self.m)
        self._finite_float("h", self.h)
        self._finite_float("n", self.n)
        self._finite_float("a", self.a)
        self._finite_float("b", self.b)
        self._positive_float("c_m", self.c_m)
        self._positive_float("dt", self.dt)
        self._finite_float("v_threshold", self.v_threshold)
        self._non_negative_float("g_na", self.g_na)
        self._non_negative_float("g_k", self.g_k)
        self._non_negative_float("g_a", self.g_a)
        self._non_negative_float("g_l", self.g_l)

    @staticmethod
    def _candidate_valid(state: tuple[float, float, float, float, float, float]) -> bool:
        v, m, h, n, a, b = state
        return (
            all(math.isfinite(x) for x in state)
            and -250.0 <= v <= 250.0
            and -0.05 <= m <= 1.05
            and -0.05 <= h <= 1.05
            and -0.05 <= n <= 1.05
            and -0.05 <= a <= 1.5
            and -0.05 <= b <= 1.05
        )

    def _derivatives(
        self, state: tuple[float, float, float, float, float, float], current: float
    ) -> tuple[float, float, float, float, float, float]:
        v, m, h, n, a, b = state
        alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n, a_inf, tau_a, b_inf, tau_b = self._rates(
            v
        )
        i_na = self.g_na * m**3 * h * (v - self.e_na)
        i_k = self.g_k * n**4 * (v - self.e_k)
        i_a = self.g_a * a**3 * b * (v - self.e_a)
        i_l = self.g_l * (v - self.e_l)
        dv = (-i_na - i_k - i_a - i_l + current) / self.c_m
        derivs = (
            dv,
            alpha_m * (1.0 - m) - beta_m * m,
            alpha_h * (1.0 - h) - beta_h * h,
            alpha_n * (1.0 - n) - beta_n * n,
            (a_inf - a) / tau_a,
            (b_inf - b) / tau_b,
        )
        if not all(math.isfinite(value) for value in derivs):
            raise FloatingPointError("Connor-Stevens derivatives must be finite")
        return derivs

    def _rk4_substep(
        self, state: tuple[float, float, float, float, float, float], current: float
    ) -> tuple[float, float, float, float, float, float]:
        dt = self.dt
        k1 = self._derivatives(state, current)
        k2_state = (
            state[0] + 0.5 * dt * k1[0],
            state[1] + 0.5 * dt * k1[1],
            state[2] + 0.5 * dt * k1[2],
            state[3] + 0.5 * dt * k1[3],
            state[4] + 0.5 * dt * k1[4],
            state[5] + 0.5 * dt * k1[5],
        )
        k2 = self._derivatives(k2_state, current)
        k3_state = (
            state[0] + 0.5 * dt * k2[0],
            state[1] + 0.5 * dt * k2[1],
            state[2] + 0.5 * dt * k2[2],
            state[3] + 0.5 * dt * k2[3],
            state[4] + 0.5 * dt * k2[4],
            state[5] + 0.5 * dt * k2[5],
        )
        k3 = self._derivatives(k3_state, current)
        k4_state = (
            state[0] + dt * k3[0],
            state[1] + dt * k3[1],
            state[2] + dt * k3[2],
            state[3] + dt * k3[3],
            state[4] + dt * k3[4],
            state[5] + dt * k3[5],
        )
        k4 = self._derivatives(k4_state, current)
        candidate = (
            state[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state[2] + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
            state[3] + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
            state[4] + dt * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]) / 6.0,
            state[5] + dt * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5]) / 6.0,
        )
        if not self._candidate_valid(candidate):
            raise FloatingPointError("Connor-Stevens RK4 candidate left finite physical bounds")
        return candidate

    def _rk4_candidate(self, current: float) -> tuple[float, float, float, float, float, float]:
        self._validate_runtime_state()
        current = self._finite_float("current", current)
        state = (self.v, self.m, self.h, self.n, self.a, self.b)
        if not self._candidate_valid(state):
            raise FloatingPointError(
                "Connor-Stevens runtime state is outside finite physical bounds"
            )
        for _ in range(int(1.0 / max(self.dt, 0.001))):
            state = self._rk4_substep(state, current)
        return state

    def step(self, current: float) -> int:
        """Advance one macro-step and return an upward-threshold spike flag.

        The method is fail-closed: invalid current, corrupted state, overflowed
        rates, or an invalid candidate raise before mutating the previous state.
        """
        v_prev = self.v
        candidate = self._rk4_candidate(current)
        self.v, self.m, self.h, self.n, self.a, self.b = candidate
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def _matches_rust_engine_contract(self) -> bool:
        """Return whether the instance matches the Rust engine default contract."""
        for name, expected in _RUST_ENGINE_DEFAULTS.items():
            if float(getattr(self, name)) != expected:
                return False
        return True

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` macro-steps, returning ``(v_trace, spikes)``.

        Rust path uses the engine under factory defaults; parity is ULP-bounded
        (not bit-identical) due to transcendental rate functions.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if backend not in ("auto", "python", "rust"):
            raise ValueError(f"backend must be auto/python/rust, got {backend!r}")
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        prefer_rust = backend == "rust" or (
            backend == "auto" and _HAS_RUST and self._matches_rust_engine_contract()
        )
        if prefer_rust:
            if not _HAS_RUST or _EngineCls is None:
                raise RuntimeError(
                    "Rust ConnorStevens backend requested but sc_neurocore_engine is unavailable."
                )
            if not self._matches_rust_engine_contract():
                raise RuntimeError(
                    "Rust ConnorStevens backend requires factory-default parameters "
                    "and initial state."
                )
            trace, spikes, state = self._simulate_rust(n_steps, current)
        else:
            if backend == "rust":
                raise RuntimeError(
                    "Rust ConnorStevens backend requested but sc_neurocore_engine is unavailable."
                )
            trace, spikes, state = self._simulate_python(n_steps, current)
        self.v, self.m, self.h, self.n, self.a, self.b = state
        return trace, spikes

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float, float, float]]:
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for t in range(n_steps):
            spikes += self.step(current)
            trace[t] = self.v
        return trace, spikes, (self.v, self.m, self.h, self.n, self.a, self.b)

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float, float, float]]:
        assert _EngineCls is not None
        neuron = _EngineCls()
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for t in range(n_steps):
            spikes += int(neuron.step(float(current)))
            st = neuron.get_state()
            trace[t] = float(st["v"])
        st = neuron.get_state()
        return (
            trace,
            spikes,
            (
                float(st["v"]),
                float(st["m"]),
                float(st["h"]),
                float(st["n"]),
                float(st["a"]),
                float(st["b"]),
            ),
        )

    def reset(self) -> None:
        self.v = -68.0
        self.m, self.h, self.n, self.a, self.b = 0.01, 0.99, 0.1, 0.5, 0.1
