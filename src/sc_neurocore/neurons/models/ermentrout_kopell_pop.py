# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Montbrió, Pazó, and Roxin 2015 exact QIF mean field

"""Source-bound Montbrió–Pazó–Roxin firing-rate equations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

_STATE_NAMES = ("r", "v")
_PARAM_NAMES = ("tau", "delta", "eta_bar", "j", "dt")

ErmentroutKopellPopulationResult = dict[str, npt.NDArray[np.float64] | float]


@dataclass
class ErmentroutKopellPopulation:
    """Represent the exact macroscopic QIF firing-rate equations.

    The public class name is retained for compatibility.  Its dynamics are
    equations (12a–b) of Montbrió, Pazó, and Roxin (2015), not the single-cell
    Ermentrout–Kopell theta model.  ``tau`` restores an explicit membrane time
    scale to the dimensionless source equations, so the maintained flow is

    ``dr/dt = delta/(pi*tau**2) + 2*r*v/tau``

    ``dv/dt = (v**2 + eta_bar + I + j*tau*r - (pi*tau*r)**2)/tau``.

    One call to :meth:`step` applies a simultaneous explicit-Euler update.
    That solver is an implementation contract; the publication specifies the
    continuous ordinary differential equations.

    Parameters
    ----------
    r : float, default=0.1
        Initial population firing rate.  It must be non-negative.
    v : float, default=-2.0
        Initial mean membrane potential.
    tau : float, default=1.0
        Positive membrane time scale.
    delta : float, default=1.0
        Non-negative half-width of the Lorentzian excitability distribution.
    eta_bar : float, default=-5.0
        Centre of the excitability distribution.
    j : float, default=15.0
        Recurrent coupling strength.
    dt : float, default=0.01
        Positive explicit-Euler step.

    References
    ----------
    Montbrió, E., Pazó, D., and Roxin, A. (2015), Physical Review X 5,
    021028. https://doi.org/10.1103/PhysRevX.5.021028
    """

    r: float = 0.1
    v: float = -2.0
    tau: float = 1.0
    delta: float = 1.0
    eta_bar: float = -5.0
    j: float = 15.0
    dt: float = 0.01

    def __post_init__(self) -> None:
        """Normalise scalar fields and reject an invalid configuration."""
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        if self.r < 0.0:
            raise ValueError("r must be non-negative")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if self.delta < 0.0:
            raise ValueError("delta must be non-negative")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")

    def _validated_state(self) -> tuple[float, float]:
        """Return the current finite physical state."""
        try:
            r = float(self.r)
            v = float(self.v)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Montbrió population state must be numeric") from exc
        if not math.isfinite(r) or not math.isfinite(v):
            raise ValueError("Montbrió population state must be finite")
        if r < 0.0:
            raise ValueError("population firing rate must be non-negative")
        return r, v

    def _validated_parameters(self) -> tuple[float, float, float, float, float]:
        """Return the current finite numerical configuration without mutation."""
        try:
            tau, delta, eta_bar, coupling, dt = (
                float(self.tau),
                float(self.delta),
                float(self.eta_bar),
                float(self.j),
                float(self.dt),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Montbrió population parameters must be numeric") from exc
        if not all(math.isfinite(value) for value in (tau, delta, eta_bar, coupling, dt)):
            raise ValueError("Montbrió population parameters must be finite")
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        if delta < 0.0:
            raise ValueError("delta must be non-negative")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        return tau, delta, eta_bar, coupling, dt

    def _next_state(self, ext_input: float) -> tuple[float, float]:
        """Compute, but do not install, one simultaneous Euler candidate."""
        try:
            drive = float(ext_input)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("external input must be numeric") from exc
        if not math.isfinite(drive):
            raise ValueError("external input must be finite")
        r, v = self._validated_state()
        tau, delta, eta_bar, coupling, dt = self._validated_parameters()
        dr = delta / (math.pi * tau**2) + 2.0 * r * v / tau
        dv = (v**2 + eta_bar + drive + coupling * tau * r - (math.pi * tau * r) ** 2) / tau
        next_r = r + dt * dr
        next_v = v + dt * dv
        if not math.isfinite(next_r) or not math.isfinite(next_v):
            raise FloatingPointError("Montbrió population candidate must remain finite")
        if next_r < 0.0:
            raise FloatingPointError("Montbrió population candidate rate became negative")
        return next_r, next_v

    def step(self, ext_input: float = 0.0) -> float:
        """Advance one Euler step atomically and return the firing rate.

        Parameters
        ----------
        ext_input : float, default=0.0
            External population drive ``I(t)`` for this Euler step.

        Returns
        -------
        float
            Post-update population firing rate ``r``.

        Raises
        ------
        ValueError
            If the input, current state, or configuration is invalid.
        FloatingPointError
            If the complete candidate is non-finite or has a negative rate.

        Notes
        -----
        Mutation is atomic: both candidates are validated before either state
        is installed.
        """
        next_r, next_v = self._next_state(ext_input)
        self.r, self.v = next_r, next_v
        return self.r

    def simulate(
        self,
        ext_input: npt.ArrayLike,
        *,
        backend: str = "auto",
    ) -> ErmentroutKopellPopulationResult:
        """Run one atomic batch through a maintained execution backend.

        Parameters
        ----------
        ext_input : ArrayLike
            One finite external drive value per simultaneous Euler step.
        backend : str, default="auto"
            ``python``, ``rust``, ``julia``, ``go``, ``mojo``, or measured
            ascending-latency selection.

        Returns
        -------
        dict[str, numpy.ndarray | float]
            Post-update ``r`` and ``v`` traces plus both final-state receipts.

        Raises
        ------
        ValueError
            If an input, state, parameter, or backend name is invalid.
        RuntimeError
            If an explicitly requested compiled backend is unavailable.
        FloatingPointError
            If a candidate or backend result violates the physical contract.

        Notes
        -----
        The object is updated only after the complete backend result passes
        validation, so rejected batches preserve both caller-visible states.
        """
        from sc_neurocore.accel.ermentrout_kopell_pop import simulate_ermentrout_kopell_pop

        result = simulate_ermentrout_kopell_pop(
            self.r,
            self.v,
            self.tau,
            self.delta,
            self.eta_bar,
            self.j,
            self.dt,
            ext_input,
            backend=backend,
        )
        self.r = float(cast(float, result["r_final"]))
        self.v = float(cast(float, result["v_final"]))
        return result

    def reset(self) -> None:
        """Restore the two dynamic states while preserving parameters."""
        self.r = 0.1
        self.v = -2.0


__all__ = ["ErmentroutKopellPopulation", "ErmentroutKopellPopulationResult"]
