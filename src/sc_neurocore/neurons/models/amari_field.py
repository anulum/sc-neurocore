# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Amari 1977 single-layer lateral-inhibition neural field

"""Finite periodic-grid specialization of Amari's 1977 neural field."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
AmariFieldResult: TypeAlias = dict[str, FloatArray]


def _finite(value: object, name: str) -> float:
    """Return one finite float or raise a field-specific ``ValueError``."""
    try:
        converted = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite real number")
    return converted


@dataclass
class AmariNeuralField:
    """Discretize Amari's homogeneous single-layer field on a periodic ring.

    The maintained equation is Amari (1977), Eq. (3), with the paper's
    source-level Heaviside output and a declared difference-of-exponentials
    lateral-inhibition kernel. ``current`` supplies the combined homogeneous
    level ``h`` and deviational input ``s(x,t)``. A scalar drive is broadcast;
    a vector drive must contain exactly ``n`` finite samples.

    One call performs a simultaneous explicit-Euler update and returns the
    mean pulse-emission rate (active-site fraction). This is a continuous
    population-rate model; the returned value is not a spike event.

    Parameters
    ----------
    n:
        Number of uniformly spaced sites on the periodic ring; at least two.
    tau:
        Positive field time constant.
    a_exc, a_width:
        Non-negative local-excitation amplitude and positive inverse width.
    b_inh, b_width:
        Non-negative distal-inhibition amplitude and positive inverse width.
    dx, dt:
        Positive spatial and temporal discretization steps.
    u:
        Optional finite initial field state of shape ``(n,)``.

    Raises
    ------
    ValueError
        If configuration, input, or a candidate state is invalid. Failed
        updates leave ``u`` unchanged.
    """

    n: int = 64
    tau: float = 10.0
    a_exc: float = 1.5
    a_width: float = 2.0
    b_inh: float = 0.75
    b_width: float = 1.0
    dx: float = 0.5
    dt: float = 0.5
    u: FloatArray | None = field(default=None, repr=False)
    _w: FloatArray = field(init=False, repr=False)
    _interaction: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration and build the circular interaction matrix."""
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 2:
            raise ValueError("n must be an integer greater than or equal to two")
        for name in ("tau", "a_exc", "a_width", "b_inh", "b_width", "dx", "dt"):
            setattr(self, name, _finite(getattr(self, name), name))
        if self.tau <= 0.0 or self.a_width <= 0.0 or self.b_width <= 0.0:
            raise ValueError("tau and kernel inverse widths must be positive")
        if self.a_exc < 0.0 or self.b_inh < 0.0:
            raise ValueError("kernel amplitudes must be non-negative")
        if self.dx <= 0.0 or self.dt <= 0.0:
            raise ValueError("dx and dt must be positive")
        if self.u is None:
            self.u = np.zeros(self.n, dtype=np.float64)
        else:
            state = np.ascontiguousarray(self.u, dtype=np.float64)
            if state.shape != (self.n,) or not np.isfinite(state).all():
                raise ValueError(f"u must be a finite vector with shape ({self.n},)")
            self.u = state.copy()
        self._build_kernel()

    def _build_kernel(self) -> None:
        """Build and validate a local-excitation/distal-inhibition ring kernel."""
        offsets = np.arange(self.n, dtype=np.float64)
        distances = np.minimum(offsets, self.n - offsets) * self.dx
        self._w = self.a_exc * np.exp(-self.a_width * distances) - self.b_inh * np.exp(
            -self.b_width * distances
        )
        if not np.isfinite(self._w).all():
            raise ValueError("interaction kernel must be finite")
        if self._w[0] <= 0.0 or self._w[self.n // 2] >= 0.0:
            raise ValueError("kernel must be locally excitatory and distally inhibitory")
        indices = np.arange(self.n)
        self._interaction = self._w[(indices[:, None] - indices[None, :]) % self.n]

    def _drive(self, current: ArrayLike) -> FloatArray:
        """Normalize one scalar or exact-length vector field stimulus."""
        drive = np.asarray(current, dtype=np.float64)
        if drive.ndim == 0:
            drive = np.full(self.n, float(drive), dtype=np.float64)
        elif drive.shape == (self.n,):
            drive = np.ascontiguousarray(drive)
        else:
            raise ValueError(f"current must be scalar or have shape ({self.n},)")
        if not np.isfinite(drive).all():
            raise ValueError("current must contain only finite values")
        return drive

    def step(self, current: ArrayLike) -> float:
        """Advance one atomic Euler step and return mean source-level activity."""
        drive = self._drive(current)
        state = self.u
        if state is None:  # pragma: no cover - dataclass invariant
            raise RuntimeError("Amari field state was not initialized")
        activity = (state > 0.0).astype(np.float64)
        convolution = (self._interaction @ activity) * self.dx
        candidate = state + (-state + convolution + drive) * (self.dt / self.tau)
        if not np.isfinite(candidate).all():
            raise ValueError("Amari field candidate state must remain finite")
        self.u = np.ascontiguousarray(candidate)
        return float(np.count_nonzero(self.u > 0.0) / self.n)

    def simulate(self, currents: ArrayLike, *, backend: str = "auto") -> AmariFieldResult:
        """Run a complete drive batch through one maintained execution lane.

        ``currents`` has shape ``(steps, n)`` or ``(steps,)`` for homogeneous
        broadcast drives. The returned mapping contains ``states``,
        ``mean_rates``, and ``final_state`` arrays. Native backend failures are
        reported; no implementation silently substitutes Python.
        """
        from sc_neurocore.accel.amari_field import simulate_amari_field

        state = self.u
        if state is None:  # pragma: no cover - dataclass invariant
            raise RuntimeError("Amari field state was not initialized")
        return simulate_amari_field(
            state,
            self.tau,
            self.a_exc,
            self.a_width,
            self.b_inh,
            self.b_width,
            self.dx,
            self.dt,
            currents,
            backend=backend,
        )

    def reset(self) -> None:
        """Zero every dynamic site while preserving numerical configuration."""
        self.u = np.zeros(self.n, dtype=np.float64)


__all__ = ["AmariFieldResult", "AmariNeuralField"]
