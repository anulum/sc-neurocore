# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Theta neuron — canonical Type-I on the unit circle

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import theta as _backends

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

    ``simulate`` exposes the Python reference and all four compiled acceleration
    lanes. The Rust engine retains its factory-default boundary; Julia, Go, and
    Mojo transport the complete phase and integration contract.
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
        """Advance a sequential trace through Python, Rust, Julia, Go, or Mojo.

        Parameters
        ----------
        n_steps:
            Number of updates (non-negative).
        current:
            Constant injected current (finite).
        backend:
            One of ``"auto"``, ``"python"``, ``"rust"``, ``"julia"``,
            ``"go"``, or ``"mojo"``. Auto probes Go, Julia, Mojo,
            compatible Rust, then Python, avoiding Julia initialisation when
            the Go shared library is available.
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
            if not _backends._HAS_RUST or _backends._EngineThetaCls is None:
                raise RuntimeError(
                    "Rust Theta backend requested but sc_neurocore_engine is unavailable."
                )
            if not self._matches_rust_engine_contract():
                raise RuntimeError(
                    "Rust Theta backend requires factory-default parameters and initial state."
                )
            trace, spikes, state = _backends.simulate_rust(n_steps, current)
        elif selected == "julia":
            if not _backends.ensure_julia_loaded():
                raise RuntimeError(
                    "Julia Theta backend requested but juliacall or the module is unavailable."
                )
            trace, spikes, state = self._simulate_full_contract(
                _backends.simulate_julia, n_steps, current
            )
        elif selected == "go":
            if not _backends.ensure_go_loaded():
                raise RuntimeError(
                    "Go Theta backend requested but libtheta.so is not built; run "
                    "go build -buildmode=c-shared -o libtheta.so theta.go in "
                    "accel/go/neurons/theta."
                )
            trace, spikes, state = self._simulate_full_contract(
                _backends.simulate_go, n_steps, current
            )
        elif selected == "mojo":
            if not _backends.ensure_mojo_loaded():
                raise RuntimeError(
                    "Mojo Theta backend requested but libtheta.so is not built; run "
                    "mojo build --emit shared-lib -o libtheta.so theta.mojo in "
                    "accel/mojo/kernels."
                )
            trace, spikes, state = self._simulate_full_contract(
                _backends.simulate_mojo, n_steps, current
            )
        else:
            trace, spikes, state = self._simulate_python(n_steps, current)
        self.theta = state
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
                [float, float, int, float],
                tuple[npt.NDArray[np.float64], int, float],
            ],
            runner,
        )
        return native(self.theta, self.dt, n_steps, current)

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for t in range(n_steps):
            spikes += self.step(current)
            trace[t] = self.theta
        return trace, spikes, self.theta

    def reset(self) -> None:
        """Restore the runtime phase while preserving the integration step."""
        self.theta = 0.0

    def _validate_runtime_state(self) -> None:
        if not math.isfinite(self.theta):
            raise ValueError("runtime phase state must be finite")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("runtime dt must be finite and positive")
