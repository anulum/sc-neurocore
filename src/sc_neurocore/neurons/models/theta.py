# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Theta neuron — canonical Type-I on the unit circle

from __future__ import annotations

import importlib as _importlib
import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

# Rust engine path: factory-default exact-flow step is bit-identical to pure Python.
try:
    _EngineCls: Optional[type[Any]] = _importlib.import_module(
        "sc_neurocore_engine"
    ).ThetaNeuron
    _HAS_RUST = True
except (ImportError, AttributeError):
    _EngineCls = None
    _HAS_RUST = False

_RUST_ENGINE_DEFAULTS: dict[str, float] = {
    "theta": 0.0,
    "dt": 0.01,
}


@dataclass
class ThetaNeuron:
    """Theta neuron — canonical Type-I on the unit circle.

    dθ/dt = (1 - cos θ) + (1 + cos θ) · I
    Spike when θ crosses π.
    Ermentrout & Kopell 1986.

    Reference: Ermentrout, G.B. & Kopell, N. (1986). SIAM J. Appl. Math. 46:233–253.

    ``simulate`` supports ``backend`` values ``python``, ``rust``, and ``auto``
    (prefer Rust under the factory-default contract when the engine is present).
    """

    theta: float = 0.0
    dt: float = 0.01

    def __post_init__(self) -> None:
        if not math.isfinite(self.theta):
            raise ValueError("theta must be finite")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.theta = self._wrap_phase(self.theta)

    def _matches_rust_engine_contract(self) -> bool:
        """Return whether the instance matches the Rust engine default contract."""
        for name, expected in _RUST_ENGINE_DEFAULTS.items():
            if float(getattr(self, name)) != expected:
                return False
        return True

    @staticmethod
    def _wrap_phase(theta: float) -> float:
        return ((theta + math.pi) % (2.0 * math.pi)) - math.pi

    def _exact_candidate(self, current: float) -> tuple[float, bool]:
        y = math.tan(self.theta / 2.0)
        if current > 0.0:
            root_i = math.sqrt(current)
            phase = math.atan(y / root_i)
            next_phase = phase + root_i * self.dt
            spiked = next_phase >= math.pi / 2.0
            if math.isclose(math.cos(next_phase), 0.0, rel_tol=0.0, abs_tol=1e-15):
                return -math.pi, spiked
            return self._wrap_phase(2.0 * math.atan(root_i * math.tan(next_phase))), spiked
        if current == 0.0:
            denominator = 1.0 - y * self.dt
            if math.isclose(denominator, 0.0, rel_tol=0.0, abs_tol=1e-15):
                return -math.pi, True
            next_y = y / denominator
            return self._wrap_phase(2.0 * math.atan(next_y)), denominator <= 0.0

        root_i = math.sqrt(-current)
        if math.isclose(y, -root_i, rel_tol=0.0, abs_tol=1e-15):
            return self.theta, False
        numerator_ratio = (y - root_i) / (y + root_i)
        try:
            evolved_ratio = numerator_ratio * math.exp(2.0 * root_i * self.dt)
        except OverflowError:
            return math.nan, False
        denominator = 1.0 - evolved_ratio
        spiked = numerator_ratio < 1.0 <= evolved_ratio or math.isclose(
            denominator,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        if spiked and math.isclose(denominator, 0.0, rel_tol=0.0, abs_tol=1e-15):
            return -math.pi, True
        next_y = root_i * (1.0 + evolved_ratio) / denominator
        return self._wrap_phase(2.0 * math.atan(next_y)), spiked

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()

        next_theta, spiked = self._exact_candidate(current)
        if not math.isfinite(next_theta):
            raise ValueError("exact-flow candidate must be finite")
        self.theta = self._wrap_phase(next_theta)
        return int(spiked)

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` updates, returning ``(theta_trace, spikes)``.

        Parameters
        ----------
        n_steps:
            Number of updates (non-negative).
        current:
            Constant injected current (finite).
        backend:
            ``python``, ``rust`` (factory defaults only), or ``auto``.
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
            if not _HAS_RUST or _EngineCls is None:
                raise RuntimeError(
                    "Rust Theta backend requested but sc_neurocore_engine is unavailable."
                )
            if not self._matches_rust_engine_contract():
                raise RuntimeError(
                    "Rust Theta backend requires factory-default parameters and initial state."
                )
            trace, spikes, state = self._simulate_rust(n_steps, current)
        else:
            if backend == "rust":
                raise RuntimeError(
                    "Rust Theta backend requested but sc_neurocore_engine is unavailable."
                )
            trace, spikes, state = self._simulate_python(n_steps, current)
        self.theta = state
        return trace, spikes

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for t in range(n_steps):
            spikes += self.step(current)
            trace[t] = self.theta
        return trace, spikes, self.theta

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        assert _EngineCls is not None
        neuron = _EngineCls()
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for t in range(n_steps):
            spikes += int(neuron.step(float(current)))
            trace[t] = float(neuron.get_state()["theta"])
        return trace, spikes, float(neuron.get_state()["theta"])

    def reset(self) -> None:
        self.theta = 0.0

    def _validate_runtime_state(self) -> None:
        if not math.isfinite(self.theta):
            raise ValueError("runtime phase state must be finite")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("runtime dt must be finite and positive")
