# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quadratic Integrate-and-Fire — canonical Type-I excitability

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
    ).QuadraticIFNeuron
    _HAS_RUST = True
except (ImportError, AttributeError):
    _EngineCls = None
    _HAS_RUST = False

_RUST_ENGINE_DEFAULTS: dict[str, float] = {
    "v": -1.0,
    "v_reset": -1.0,
    "v_peak": 1.0,
    "dt": 0.01,
}


@dataclass
class QuadraticIFNeuron:
    """Quadratic Integrate-and-Fire — canonical Type-I excitability.

    dv/dt = v² + I
    Reset when v >= v_peak.

    Reference: Latham, P.E. et al. (2000). J. Neurophysiol. 83:808–827.

    ``simulate`` supports ``backend`` values ``python``, ``rust``, and ``auto``
    (prefer Rust under the factory-default contract when the engine is present).
    """

    v: float = -1.0
    v_reset: float = -1.0
    v_peak: float = 1.0
    dt: float = 0.01

    def __post_init__(self) -> None:
        for field in ("v", "v_reset", "v_peak"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if self.v >= self.v_peak:
            raise ValueError("v must be below v_peak")
        if self.v_reset >= self.v_peak:
            raise ValueError("v_peak must be greater than v_reset")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")

    def _matches_rust_engine_contract(self) -> bool:
        """Return whether the instance matches the Rust engine default contract."""
        for name, expected in _RUST_ENGINE_DEFAULTS.items():
            if float(getattr(self, name)) != expected:
                return False
        return True

    def _exact_candidate(self, current: float) -> tuple[float, bool]:
        if current > 0.0:
            root_i = math.sqrt(current)
            phase = math.atan(self.v / root_i)
            peak_phase = math.atan(self.v_peak / root_i)
            next_phase = phase + root_i * self.dt
            if next_phase >= peak_phase or next_phase >= math.pi / 2.0:
                return self.v_reset, True
            return root_i * math.tan(next_phase), False
        if current == 0.0:
            denominator = 1.0 - self.v * self.dt
            if denominator <= 0.0:
                return self.v_reset, True
            next_v = self.v / denominator
            return (self.v_reset, True) if next_v >= self.v_peak else (next_v, False)

        root_i = math.sqrt(-current)
        if math.isclose(self.v, -root_i, rel_tol=0.0, abs_tol=1e-15):
            return self.v, False
        numerator_ratio = (self.v - root_i) / (self.v + root_i)
        try:
            evolved_ratio = numerator_ratio * math.exp(2.0 * root_i * self.dt)
        except OverflowError:
            return math.nan, False
        denominator = 1.0 - evolved_ratio
        if numerator_ratio < 1.0 <= evolved_ratio or math.isclose(
            denominator, 0.0, rel_tol=0.0, abs_tol=1e-15
        ):
            return self.v_reset, True
        next_v = root_i * (1.0 + evolved_ratio) / denominator
        return (self.v_reset, True) if next_v >= self.v_peak else (next_v, False)

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        next_v, spiked = self._exact_candidate(current)
        if not math.isfinite(next_v):
            raise ValueError("exact-flow candidate must be finite")
        self.v = next_v
        return int(spiked)

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` updates, returning ``(v_trace, spikes)``.

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
        if not math.isfinite(self.v):
            raise ValueError("runtime v must be finite")

        prefer_rust = backend == "rust" or (
            backend == "auto" and _HAS_RUST and self._matches_rust_engine_contract()
        )
        if prefer_rust:
            if not _HAS_RUST or _EngineCls is None:
                raise RuntimeError(
                    "Rust QuadraticIF backend requested but sc_neurocore_engine is unavailable."
                )
            if not self._matches_rust_engine_contract():
                raise RuntimeError(
                    "Rust QuadraticIF backend requires factory-default parameters "
                    "and initial state."
                )
            trace, spikes, state_v = self._simulate_rust(n_steps, current)
        else:
            if backend == "rust":
                raise RuntimeError(
                    "Rust QuadraticIF backend requested but sc_neurocore_engine is unavailable."
                )
            trace, spikes, state_v = self._simulate_python(n_steps, current)
        self.v = state_v
        return trace, spikes

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for t in range(n_steps):
            spikes += self.step(current)
            trace[t] = self.v
        return trace, spikes, self.v

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        assert _EngineCls is not None
        neuron = _EngineCls()
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for t in range(n_steps):
            spikes += int(neuron.step(float(current)))
            trace[t] = float(neuron.get_state()["v"])
        return trace, spikes, float(neuron.get_state()["v"])

    def reset(self) -> None:
        self.v = self.v_reset
