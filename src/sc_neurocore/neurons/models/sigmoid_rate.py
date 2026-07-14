# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exact-relaxation continuous rate model with sigmoid transfer

"""Validated single-population rate relaxation inspired by Wilson and Cowan.

The 1972 source derives coupled excitatory and inhibitory population dynamics.
This maintained scalar model deliberately isolates the shared first-order
relaxation-to-sigmoid motif; it is not the full coupled Wilson-Cowan system.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import SupportsIndex, cast

import numpy as np
import numpy.typing as npt

_MAX_STEPS = (1 << 31) - 1
_BACKENDS = ("auto", "python", "rust", "julia", "go", "mojo")


def _step_count(value: object) -> int:
    """Return a C-ABI-safe non-negative batch length."""
    if isinstance(value, bool):
        raise ValueError("n_steps must be a non-negative integer")
    try:
        converted = operator.index(cast(SupportsIndex, value))
    except TypeError as exc:
        raise ValueError("n_steps must be a non-negative integer") from exc
    result = int(converted)
    if not 0 <= result <= _MAX_STEPS:
        raise ValueError(f"n_steps must be in [0, {_MAX_STEPS}]")
    return result


@dataclass
class SigmoidRateNeuron:
    """Scalar rate relaxation with a stable logistic transfer.

    tau dr/dt = -r + sigma(beta * (input - theta))

    Wilson and Cowan (1972) derive the coupled population framework that
    motivates this reduced single-unit motif. The complete excitatory and
    inhibitory model is represented separately by ``WilsonCowanUnit``.
    """

    r: float = 0.0
    tau: float = 10.0
    beta: float = 1.0
    theta: float = 0.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        for field in ("beta", "theta"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if not math.isfinite(self.r):
            raise ValueError("r must be finite")
        if not 0.0 <= self.r <= 1.0:
            raise ValueError("r must be in [0, 1]")
        for field in ("tau", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")

    def step(self, current: float) -> float:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()

        sigma = self._stable_sigmoid(self.beta, current, self.theta)
        next_r = self._exact_relaxation(self.r, sigma)
        self.r = next_r
        return next_r

    def simulate(
        self,
        n_steps: int,
        current: float = 3.0,
        backend: str = "auto",
    ) -> npt.NDArray[np.float64]:
        """Return a post-step rate trace through one maintained backend.

        A successful batch atomically commits only the final rate. Explicitly
        requested unavailable backends fail instead of silently falling back.
        """
        steps = _step_count(n_steps)
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be auto/python/rust/julia/go/mojo, got {backend!r}")

        from sc_neurocore.accel import sigmoid_rate as backends

        selected = backends.auto_backend() if backend == "auto" else backend
        if selected == "python":
            trace, final_rate = self._simulate_python(steps, current)
        else:
            if not backends.backend_available(selected):
                raise RuntimeError(f"{selected.title()} SigmoidRate backend is unavailable.")
            runner = {
                "rust": backends.simulate_rust,
                "julia": backends.simulate_julia,
                "go": backends.simulate_go,
                "mojo": backends.simulate_mojo,
            }[selected]
            trace, final_rate = runner(
                self.r,
                self.tau,
                self.beta,
                self.theta,
                self.dt,
                steps,
                current,
            )

        trace, final_rate = backends.normalise_result(
            trace,
            final_rate,
            n_steps=steps,
            initial_rate=self.r,
        )
        self.r = final_rate
        return trace

    def _simulate_python(
        self,
        n_steps: int,
        current: float,
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Evaluate an atomic local batch without mutating the instance."""
        target = self._stable_sigmoid(self.beta, current, self.theta)
        decay = math.exp(-self.dt / self.tau)
        rate = self.r
        trace = np.empty(n_steps, dtype=np.float64)
        for index in range(n_steps):
            rate = decay * rate + (1.0 - decay) * target
            trace[index] = rate
        return trace, rate

    def reset(self) -> None:
        """Restore the dynamic rate without changing configuration."""
        self.r = 0.0

    def _validate_runtime_state(self) -> None:
        if not (
            math.isfinite(self.r)
            and math.isfinite(self.beta)
            and math.isfinite(self.theta)
            and math.isfinite(self.tau)
            and math.isfinite(self.dt)
        ):
            raise ValueError("runtime rate state must be finite")
        if not 0.0 <= self.r <= 1.0:
            raise ValueError("runtime rate state must be in [0, 1]")
        if self.tau <= 0.0 or self.dt <= 0.0:
            raise ValueError("runtime time constants must be positive")

    def _exact_relaxation(self, rate: float, steady_state: float) -> float:
        decay = math.exp(-self.dt / self.tau)
        return decay * rate + (1.0 - decay) * steady_state

    @staticmethod
    def _stable_sigmoid(beta: float, current: float, theta: float) -> float:
        delta = current - theta
        z = beta * delta
        if math.isinf(z):
            return 1.0 if z > 0.0 else 0.0
        if not math.isfinite(z):
            raise ValueError("sigmoid argument must remain finite or saturating")
        if z >= 0.0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)
