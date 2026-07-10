# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Non-leaky integrate-and-fire. Lapicque 1907 (no leak)

from __future__ import annotations

import importlib as _importlib
import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

# Rust engine path: factory-default Euler step is bit-identical to pure NumPy.
try:
    _EngineCls: Optional[type[Any]] = _importlib.import_module(
        "sc_neurocore_engine"
    ).PerfectIntegratorNeuron
    _HAS_RUST = True
except (ImportError, AttributeError):
    _EngineCls = None
    _HAS_RUST = False

_RUST_ENGINE_DEFAULTS: dict[str, float] = {
    "v": 0.0,
    "c_m": 1.0,
    "v_threshold": 1.0,
    "v_reset": 0.0,
    "dt": 0.1,
}


@dataclass
class PerfectIntegratorNeuron:
    """Non-leaky integrate-and-fire. Lapicque 1907 (no leak).

    dV/dt = I / C

    Reference: Gerstner, W. et al. (2014). Neuronal Dynamics. Cambridge Univ. Press, §1.3.

    ``simulate`` supports ``backend`` values ``python``, ``rust``, and ``auto``
    (prefer Rust under the factory-default contract when the engine is present).
    """

    v: float = 0.0
    c_m: float = 1.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        for field in ("v", "v_threshold", "v_reset"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if self.v_threshold <= self.v_reset:
            raise ValueError("v_threshold must be greater than v_reset")
        if self.v >= self.v_threshold:
            raise ValueError("v must be below v_threshold")
        for field in ("c_m", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")

    def _matches_rust_engine_contract(self) -> bool:
        """Return whether the instance matches the Rust engine default contract."""
        for name, expected in _RUST_ENGINE_DEFAULTS.items():
            if float(getattr(self, name)) != expected:
                return False
        return True

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()
        voltage_increment = current / self.c_m * self.dt
        next_v = self.v + voltage_increment
        if not math.isfinite(voltage_increment) or not math.isfinite(next_v):
            raise ValueError("voltage increment must be finite")
        self.v = next_v
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

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
        self._validate_runtime_state()

        prefer_rust = backend == "rust" or (
            backend == "auto" and _HAS_RUST and self._matches_rust_engine_contract()
        )
        if prefer_rust:
            if not _HAS_RUST or _EngineCls is None:
                raise RuntimeError(
                    "Rust PerfectIntegrator backend requested but "
                    "sc_neurocore_engine is unavailable."
                )
            if not self._matches_rust_engine_contract():
                raise RuntimeError(
                    "Rust PerfectIntegrator backend requires factory-default "
                    "parameters and initial state."
                )
            trace, spikes, state_v = self._simulate_rust(n_steps, current)
        else:
            if backend == "rust":
                raise RuntimeError(
                    "Rust PerfectIntegrator backend requested but "
                    "sc_neurocore_engine is unavailable."
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

    def _validate_runtime_state(self) -> None:
        for field in ("v", "v_threshold", "v_reset", "c_m", "dt"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"runtime {field} must be finite")
        if self.c_m <= 0.0 or self.dt <= 0.0:
            raise ValueError("runtime c_m and dt must be positive")
        if self.v_threshold <= self.v_reset:
            raise ValueError("runtime v_threshold must be greater than v_reset")
        if self.v >= self.v_threshold:
            raise ValueError("runtime v must be below v_threshold before integration")
