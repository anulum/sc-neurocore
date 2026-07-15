# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Jansen and Rit 1995 cortical-column neural mass

"""Publication-aligned Jansen–Rit six-state neural-mass dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

_STATE_NAMES = ("y0", "y3", "y1", "y4", "y2", "y5")
_PARAM_NAMES = ("a_exc", "b_exc", "a_rate", "b_rate", "c", "e0", "v0", "r", "dt")
_STRICTLY_POSITIVE_PARAMS = ("a_exc", "b_exc", "a_rate", "b_rate", "e0", "r", "dt")

JansenRitResult = dict[str, npt.NDArray[np.float64] | float]


@dataclass
class JansenRitUnit:
    """Represent one Jansen–Rit cortical-column neural mass.

    The six first-order states implement equation (6) from Jansen and Rit
    (1995).  ``e0`` is half the maximum firing rate because :meth:`_sigmoid`
    returns ``2 * e0 / (1 + exp(r * (v0 - v)))``.  The default explicit-Euler
    step is 0.1 ms, matching the pinned Brian2 implementation used for the
    source-bound trace; the continuous equations do not prescribe a solver.

    Parameters
    ----------
    y0, y3, y1, y4, y2, y5 : float, default=0.0
        Initial postsynaptic-potential states and their first derivatives.
    a_exc, b_exc : float
        Excitatory and inhibitory synaptic gains ``A`` and ``B`` in mV.
    a_rate, b_rate : float
        Excitatory and inhibitory inverse time constants in s⁻¹.
    c : float
        Base connectivity ``C1``.  Derived couplings are ``C2=0.8*C1`` and
        ``C3=C4=0.25*C1``.
    e0, v0, r : float
        Sigmoid half-maximum rate, midpoint, and slope.
    dt : float, default=0.0001
        Explicit-Euler step in seconds.

    References
    ----------
    Jansen, B. H. and Rit, V. G. (1995), Biological Cybernetics 73, 357–366.
    https://doi.org/10.1007/BF00199471
    """

    y0: float = 0.0
    y3: float = 0.0
    y1: float = 0.0
    y4: float = 0.0
    y2: float = 0.0
    y5: float = 0.0
    a_exc: float = 3.25
    b_exc: float = 22.0
    a_rate: float = 100.0
    b_rate: float = 50.0
    c: float = 135.0
    e0: float = 2.5
    v0: float = 6.0
    r: float = 0.56
    dt: float = 0.0001

    def __post_init__(self) -> None:
        """Normalise scalar fields and reject an invalid configuration."""
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        for name in _STRICTLY_POSITIVE_PARAMS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.c < 0.0:
            raise ValueError("c must be non-negative")

    @staticmethod
    def _require_finite(name: str, value: float) -> float:
        out = float(value)
        if not math.isfinite(out):
            raise ValueError(f"{name} must be finite")
        return out

    def _validate_state(self, values: tuple[float, ...] | None = None) -> tuple[float, ...]:
        state = (
            values if values is not None else tuple(getattr(self, name) for name in _STATE_NAMES)
        )
        if len(state) != len(_STATE_NAMES):
            raise ValueError("Jansen–Rit state vector has invalid dimension")
        return tuple(
            self._require_finite(name, value)
            for name, value in zip(_STATE_NAMES, state, strict=True)
        )

    def _sigmoid(self, voltage: float) -> float:
        """Return the overflow-stable population firing-rate response."""
        drive = self._require_finite("sigmoid input", voltage)
        exponent = self.r * (self.v0 - drive)
        if exponent >= 0.0:
            exp_neg = math.exp(-exponent)
            return 2.0 * self.e0 * exp_neg / (1.0 + exp_neg)
        return 2.0 * self.e0 / (1.0 + math.exp(exponent))

    def _next_state(self, p_ext: float) -> tuple[float, ...]:
        """Compute, but do not install, one equation-(6) Euler candidate."""
        drive = self._require_finite("p_ext", p_ext)
        y0, y3, y1, y4, y2, y5 = self._validate_state()
        c1 = self.c
        c2 = 0.8 * c1
        c3 = 0.25 * c1
        c4 = 0.25 * c1
        s_pyramidal = self._sigmoid(y1 - y2)
        s_excitatory = self._sigmoid(c1 * y0)
        s_inhibitory = self._sigmoid(c3 * y0)

        dy0 = y3
        dy3 = self.a_exc * self.a_rate * s_pyramidal - 2.0 * self.a_rate * y3 - self.a_rate**2 * y0
        dy1 = y4
        dy4 = (
            self.a_exc * self.a_rate * (drive + c2 * s_excitatory)
            - 2.0 * self.a_rate * y4
            - self.a_rate**2 * y1
        )
        dy2 = y5
        dy5 = (
            self.b_exc * self.b_rate * c4 * s_inhibitory
            - 2.0 * self.b_rate * y5
            - self.b_rate**2 * y2
        )
        return self._validate_state(
            (
                y0 + dy0 * self.dt,
                y3 + dy3 * self.dt,
                y1 + dy1 * self.dt,
                y4 + dy4 * self.dt,
                y2 + dy2 * self.dt,
                y5 + dy5 * self.dt,
            )
        )

    def step(self, p_ext: float = 220.0) -> float:
        """Advance one explicit-Euler step and return the post-update EEG proxy.

        Parameters
        ----------
        p_ext : float, default=220.0
            External pulse-density drive in pulses per second.

        Returns
        -------
        float
            Post-update pyramidal potential difference ``y1 - y2`` in mV.

        Raises
        ------
        ValueError
            If the input, current state, or complete candidate is non-finite.

        Notes
        -----
        Mutation is atomic: all six candidate states are validated first.
        """
        candidate = self._next_state(p_ext)
        self.y0, self.y3, self.y1, self.y4, self.y2, self.y5 = candidate
        return self.y1 - self.y2

    def simulate(
        self,
        p_ext: npt.ArrayLike,
        *,
        backend: str = "auto",
    ) -> JansenRitResult:
        """Run an atomic batch on one maintained execution backend.

        Parameters
        ----------
        p_ext : ArrayLike
            One finite external drive per Euler step.
        backend : str, default="auto"
            ``python``, ``rust``, ``julia``, ``go``, ``mojo``, or measured
            ascending-latency selection.

        Returns
        -------
        dict[str, numpy.ndarray | float]
            Six post-update state traces, the EEG trace, and six final states.

        Raises
        ------
        ValueError
            If an input, state, parameter, or backend name is invalid.
        RuntimeError
            If an explicitly requested compiled backend is unavailable.
        FloatingPointError
            If a backend returns a malformed or inconsistent result.
        """
        from sc_neurocore.accel.jansen_rit import simulate_jansen_rit

        result = simulate_jansen_rit(
            self.y0,
            self.y3,
            self.y1,
            self.y4,
            self.y2,
            self.y5,
            self.a_exc,
            self.b_exc,
            self.a_rate,
            self.b_rate,
            self.c,
            self.e0,
            self.v0,
            self.r,
            self.dt,
            p_ext,
            backend=backend,
        )
        self.y0 = float(cast(float, result["y0_final"]))
        self.y3 = float(cast(float, result["y3_final"]))
        self.y1 = float(cast(float, result["y1_final"]))
        self.y4 = float(cast(float, result["y4_final"]))
        self.y2 = float(cast(float, result["y2_final"]))
        self.y5 = float(cast(float, result["y5_final"]))
        return result

    def reset(self) -> None:
        """Restore all six dynamic states while preserving parameters."""
        self.y0 = self.y3 = self.y1 = self.y4 = self.y2 = self.y5 = 0.0


__all__ = ["JansenRitResult", "JansenRitUnit"]
