# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lapicque 1907 — classical RC integrate-and-fire

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import lapicque as _backends

_RUST_ENGINE_DEFAULTS: dict[str, float] = {
    "v": 0.0,
    "v_rest": 0.0,
    "v_reset": 0.0,
    "v_threshold": 1.0,
    "tau": 20.0,
    "resistance": 1.0,
    "dt": 1.0,
}


@dataclass
class LapicqueNeuron:
    """Lapicque 1907 — classical RC integrate-and-fire.

    tau * dv/dt = -(v - v_rest) + R * I

    Constant-current steps use the exact RC flow:
    V(t + dt) = V_inf + (V(t) - V_inf) * exp(-dt / tau)
    where V_inf = V_rest + R * I.

    Reference: Lapicque, L. (1907). J. Physiol. Pathol. Gén. 9:620–635.

    ``simulate`` exposes the Python reference and all four compiled acceleration
    lanes. The Rust engine retains its factory-default boundary; Julia, Go, and
    Mojo transport the complete numeric state and parameter contract.
    """

    v: float = 0.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_threshold: float = 1.0
    tau: float = 20.0
    resistance: float = 1.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        self._validate_runtime_state()

    def _matches_rust_engine_contract(self) -> bool:
        """Return whether the instance matches the Rust engine default contract."""
        for name, expected in _RUST_ENGINE_DEFAULTS.items():
            if float(getattr(self, name)) != expected:
                return False
        return True

    def _validate_runtime_state(self) -> None:
        for field in ("v", "v_rest", "v_reset", "v_threshold"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if self.v_threshold <= self.v_rest:
            raise ValueError("v_threshold must be greater than v_rest")
        if self.v_threshold <= self.v_reset:
            raise ValueError("v_threshold must be greater than v_reset")
        if self.v >= self.v_threshold:
            raise ValueError("v must be below v_threshold")
        for field in ("tau", "resistance", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()
        v_inf = self.v_rest + self.resistance * current
        decay = math.exp(-self.dt / self.tau)
        next_v = v_inf + (self.v - v_inf) * decay
        if not math.isfinite(v_inf) or not math.isfinite(decay) or not math.isfinite(next_v):
            raise ValueError("voltage candidate must be finite")
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
            Non-negative number of sequential exact-flow updates.
        current:
            Finite constant current for every update.
        backend:
            One of ``"auto"``, ``"python"``, ``"rust"``, ``"julia"``,
            ``"go"``, or ``"mojo"``. Auto uses the committed measured order
            Mojo, Julia, Go, compatible Rust, then Python.

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
            if not _backends._HAS_RUST or _backends._EngineLapicqueCls is None:
                raise RuntimeError(
                    "Rust Lapicque backend requested but sc_neurocore_engine is unavailable."
                )
            if not self._matches_rust_engine_contract():
                raise RuntimeError(
                    "Rust Lapicque backend requires factory-default parameters and initial state."
                )
            trace, spikes, state_v = _backends.simulate_rust(n_steps, current)
        elif selected == "julia":
            if not _backends.ensure_julia_loaded():
                raise RuntimeError(
                    "Julia Lapicque backend requested but juliacall or the module is unavailable."
                )
            trace, spikes, state_v = self._simulate_full_contract(
                _backends.simulate_julia, n_steps, current
            )
        elif selected == "go":
            if not _backends.ensure_go_loaded():
                raise RuntimeError(
                    "Go Lapicque backend requested but liblapicque.so is not built; run "
                    "go build -buildmode=c-shared -o liblapicque.so lapicque.go in "
                    "accel/go/neurons/lapicque."
                )
            trace, spikes, state_v = self._simulate_full_contract(
                _backends.simulate_go, n_steps, current
            )
        elif selected == "mojo":
            if not _backends.ensure_mojo_loaded():
                raise RuntimeError(
                    "Mojo Lapicque backend requested but liblapicque.so is not built; run "
                    "mojo build --emit shared-lib -o liblapicque.so lapicque.mojo in "
                    "accel/mojo/kernels."
                )
            trace, spikes, state_v = self._simulate_full_contract(
                _backends.simulate_mojo, n_steps, current
            )
        else:
            trace, spikes, state_v = self._simulate_python(n_steps, current)
        self.v = state_v
        return trace, spikes

    def _simulate_full_contract(
        self,
        runner: object,
        n_steps: int,
        current: float,
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        """Pass the complete maintained numeric contract to a native runner."""
        from collections.abc import Callable
        from typing import cast

        native = cast(
            Callable[
                [float, float, float, float, float, float, float, int, float],
                tuple[npt.NDArray[np.float64], int, float],
            ],
            runner,
        )
        return native(
            self.v,
            self.v_rest,
            self.v_reset,
            self.v_threshold,
            self.tau,
            self.resistance,
            self.dt,
            n_steps,
            current,
        )

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for t in range(n_steps):
            spikes += self.step(current)
            trace[t] = self.v
        return trace, spikes, self.v

    def reset(self) -> None:
        """Restore the membrane voltage to the maintained resting state."""
        self.v = self.v_rest
