# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Validated threshold-linear population-rate transfer

"""Memoryless threshold-linear gain function for continuous population rates.

The maintained transfer is the piecewise-linear gain function used in
Gerstner et al. (2014), Eq. 18.23, with explicit threshold and gain:
``r = gain * max(0, current - theta)``. ``r`` caches the latest output; it is
not an integrated state and a positive value is not a binary spike event.
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
class ThresholdLinearRateNeuron:
    """Threshold-linear continuous-rate transfer with cached output.

    Parameters
    ----------
    r:
        Initial cached output rate. It must be finite and non-negative.
    theta:
        Finite input threshold.
    gain:
        Finite, non-negative slope above threshold.

    Notes
    -----
    Each call evaluates ``gain * max(0, current - theta)`` directly. No time
    integration or hidden history is part of this model.
    """

    r: float = 0.0
    theta: float = 0.0
    gain: float = 1.0

    def __post_init__(self) -> None:
        self._validate_runtime_state()

    def step(self, current: float) -> float:
        """Evaluate one finite input and atomically cache the resulting rate."""
        self._validate_runtime_state()
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        next_r = self._transfer(self.theta, self.gain, current)
        self.r = next_r
        return next_r

    def simulate(
        self,
        n_steps: int,
        current: float = 3.0,
        backend: str = "auto",
    ) -> npt.NDArray[np.float64]:
        """Return a post-evaluation rate trace through one maintained backend.

        The batch commits ``r`` only after the selected backend returns a
        well-formed complete trace. Explicit unavailable backends fail closed.
        """
        steps = _step_count(n_steps)
        self._validate_runtime_state()
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be auto/python/rust/julia/go/mojo, got {backend!r}")

        from sc_neurocore.accel import threshold_linear_rate as backends

        selected = backends.auto_backend() if backend == "auto" else backend
        if selected == "python":
            trace, final_rate = self._simulate_python(steps, current)
        else:
            if not backends.backend_available(selected):
                raise RuntimeError(
                    f"{selected.title()} ThresholdLinearRate backend is unavailable."
                )
            runner = {
                "rust": backends.simulate_rust,
                "julia": backends.simulate_julia,
                "go": backends.simulate_go,
                "mojo": backends.simulate_mojo,
            }[selected]
            trace, final_rate = runner(self.r, self.theta, self.gain, steps, current)

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
        """Evaluate an atomic local batch without mutating this instance."""
        rate = self._transfer(self.theta, self.gain, current)
        trace = np.full(n_steps, rate, dtype=np.float64)
        final_rate = self.r if n_steps == 0 else rate
        return trace, final_rate

    def reset(self) -> None:
        """Clear the cached output while preserving threshold and gain."""
        self.r = 0.0

    def _validate_runtime_state(self) -> None:
        """Reject invalid mutable state before evaluating an input."""
        if not math.isfinite(self.r) or self.r < 0.0:
            raise ValueError("r must be finite and non-negative")
        if not math.isfinite(self.theta):
            raise ValueError("theta must be finite")
        if not math.isfinite(self.gain) or self.gain < 0.0:
            raise ValueError("gain must be finite and non-negative")

    @staticmethod
    def _transfer(theta: float, gain: float, current: float) -> float:
        """Return one validated threshold-linear transfer value."""
        next_r = gain * max(0.0, current - theta)
        if not math.isfinite(next_r) or next_r < 0.0:
            raise ValueError("rate output must remain finite and non-negative")
        return next_r
