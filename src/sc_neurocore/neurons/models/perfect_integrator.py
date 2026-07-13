# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Non-leaky integrate-and-fire. Lapicque 1907 (no leak)

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import perfect_integrator as _backends

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

    ``simulate`` exposes the Python reference and all four compiled acceleration
    lanes. The Rust engine retains its factory-default boundary; Julia, Go, and
    Mojo transport the complete numeric state and parameter contract.
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
        """Advance a sequential trace through Python, Rust, Julia, Go, or Mojo.

        Parameters
        ----------
        n_steps:
            Non-negative number of sequential Euler updates.
        current:
            Constant injected current (finite).
        backend:
            One of ``"auto"``, ``"python"``, ``"rust"``, ``"julia"``,
            ``"go"``, or ``"mojo"``. Auto uses the committed measured order.

        Returns
        -------
        tuple[numpy.ndarray, int]
            Contiguous post-step voltage trace and total spike count.

        Raises
        ------
        ValueError
            If the step count, current, backend, or runtime state is invalid.
        RuntimeError
            If a requested compiled backend is unavailable or incompatible.
        FloatingPointError
            If a C-ABI backend rejects the numeric contract.
        """
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 0:
            raise ValueError("n_steps must be a non-negative integer")
        if backend not in ("auto", "python", "rust", "julia", "go", "mojo"):
            raise ValueError(f"backend must be auto/python/rust/julia/go/mojo, got {backend!r}")
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()

        selected = backend
        if selected == "auto":
            if _backends.ensure_mojo_loaded():
                selected = "mojo"
            elif _backends.ensure_julia_loaded():
                selected = "julia"
            elif _backends.ensure_go_loaded():
                selected = "go"
            elif _backends._HAS_RUST and self._matches_rust_engine_contract():
                selected = "rust"
            else:
                selected = "python"

        if selected == "rust":
            if not _backends._HAS_RUST or _backends._EnginePerfectIntegratorCls is None:
                raise RuntimeError(
                    "Rust PerfectIntegrator backend requested but "
                    "sc_neurocore_engine is unavailable."
                )
            if not self._matches_rust_engine_contract():
                raise RuntimeError(
                    "Rust PerfectIntegrator backend requires factory-default "
                    "parameters and initial state."
                )
            trace, spikes, state_v = _backends.simulate_rust(n_steps, current)
        elif selected == "julia":
            if not _backends.ensure_julia_loaded():
                raise RuntimeError(
                    "Julia PerfectIntegrator backend requested but juliacall or the "
                    "module is unavailable."
                )
            trace, spikes, state_v = self._simulate_full_contract(
                _backends.simulate_julia, n_steps, current
            )
        elif selected == "go":
            if not _backends.ensure_go_loaded():
                raise RuntimeError(
                    "Go PerfectIntegrator backend requested but libperfect_integrator.so "
                    "is not built; run go build -buildmode=c-shared -o "
                    "libperfect_integrator.so perfect_integrator.go in "
                    "accel/go/neurons/perfect_integrator."
                )
            trace, spikes, state_v = self._simulate_full_contract(
                _backends.simulate_go, n_steps, current
            )
        elif selected == "mojo":
            if not _backends.ensure_mojo_loaded():
                raise RuntimeError(
                    "Mojo PerfectIntegrator backend requested but "
                    "libperfect_integrator.so is not built; run mojo build --emit "
                    "shared-lib -o libperfect_integrator.so perfect_integrator.mojo in "
                    "accel/mojo/kernels."
                )
            trace, spikes, state_v = self._simulate_full_contract(
                _backends.simulate_mojo, n_steps, current
            )
        else:
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

    def _simulate_full_contract(
        self,
        runner: object,
        n_steps: int,
        current: float,
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        """Pass the complete maintained numeric contract to a native runner."""
        native = cast(
            Callable[
                [float, float, float, float, float, int, float],
                tuple[npt.NDArray[np.float64], int, float],
            ],
            runner,
        )
        return native(
            self.v,
            self.c_m,
            self.v_threshold,
            self.v_reset,
            self.dt,
            n_steps,
            current,
        )

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
