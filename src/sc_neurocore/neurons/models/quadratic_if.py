# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quadratic Integrate-and-Fire — canonical Type-I excitability

"""Exact-flow Quadratic Integrate-and-Fire dynamics and public dispatch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import quadratic_if as _backends

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

    ``simulate`` exposes the Python reference and all four compiled acceleration
    lanes. The Rust engine retains its factory-default boundary; Julia, Go, and
    Mojo transport the complete numeric state and parameter contract.
    """

    v: float = -1.0
    v_reset: float = -1.0
    v_peak: float = 1.0
    dt: float = 0.01

    def __post_init__(self) -> None:
        """Validate the finite ordered state and integration contract."""
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
        """Advance one exact constant-current Riccati-flow update.

        Parameters
        ----------
        current:
            Finite constant drive over this update.

        Returns
        -------
        int
            One when the within-step flow reaches ``v_peak``; otherwise zero.
        """
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
        """Advance a sequential trace through Python, Rust, Julia, Go, or Mojo.

        Parameters
        ----------
        n_steps:
            Non-negative number of sequential exact-flow updates.
        current:
            Constant injected current (finite).
        backend:
            One of ``"auto"``, ``"python"``, ``"rust"``, ``"julia"``,
            ``"go"``, or ``"mojo"``. Auto uses the committed production order
            Go, Julia, Mojo, compatible Rust, then Python; the Go shared library
            avoids Julia runtime initialisation on the first call.
        """
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 0:
            raise ValueError("n_steps must be a non-negative integer")
        if backend not in ("auto", "python", "rust", "julia", "go", "mojo"):
            raise ValueError(f"backend must be auto/python/rust/julia/go/mojo, got {backend!r}")
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self.__post_init__()

        selected = backend
        if selected == "auto":
            if _backends.ensure_go_loaded():
                selected = "go"
            elif _backends.ensure_julia_loaded():
                selected = "julia"
            elif _backends.ensure_mojo_loaded():
                selected = "mojo"
            elif _backends._HAS_RUST and self._matches_rust_engine_contract():
                selected = "rust"
            else:
                selected = "python"

        if selected == "rust":
            if not _backends._HAS_RUST or _backends._EngineQuadraticIFCls is None:
                raise RuntimeError(
                    "Rust QuadraticIF backend requested but sc_neurocore_engine is unavailable."
                )
            if not self._matches_rust_engine_contract():
                raise RuntimeError(
                    "Rust QuadraticIF backend requires factory-default parameters "
                    "and initial state."
                )
            trace, spikes, state_v = _backends.simulate_rust(n_steps, current)
        elif selected == "julia":
            if not _backends.ensure_julia_loaded():
                raise RuntimeError(
                    "Julia QuadraticIF backend requested but juliacall or the module is "
                    "unavailable."
                )
            trace, spikes, state_v = self._simulate_full_contract(
                _backends.simulate_julia, n_steps, current
            )
        elif selected == "go":
            if not _backends.ensure_go_loaded():
                raise RuntimeError(
                    "Go QuadraticIF backend requested but libquadratic_if.so is not built; "
                    "run go build -buildmode=c-shared -o libquadratic_if.so "
                    "quadratic_if.go in accel/go/neurons/quadratic_if."
                )
            trace, spikes, state_v = self._simulate_full_contract(
                _backends.simulate_go, n_steps, current
            )
        elif selected == "mojo":
            if not _backends.ensure_mojo_loaded():
                raise RuntimeError(
                    "Mojo QuadraticIF backend requested but libquadratic_if.so is not built; "
                    "run mojo build --emit shared-lib -o libquadratic_if.so "
                    "quadratic_if.mojo in accel/mojo/kernels."
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
        """Pass every maintained numeric field to a native runner."""
        native = cast(
            Callable[
                [float, float, float, float, int, float],
                tuple[npt.NDArray[np.float64], int, float],
            ],
            runner,
        )
        return native(
            self.v,
            self.v_reset,
            self.v_peak,
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
        """Restore the runtime voltage while preserving configured parameters."""
        self.v = self.v_reset
