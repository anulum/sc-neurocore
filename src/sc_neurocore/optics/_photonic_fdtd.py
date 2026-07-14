# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Photonic FDTD solvers

"""One- and two-dimensional finite-difference photonic reference solvers."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np
import numpy.typing as npt

from ._photonic_types import _require_finite, _require_non_negative, _require_positive


def _require_count(value: int, name: str, minimum: int = 0) -> None:
    """Reject a Boolean, non-integer, or undersized count."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


class FDTDSolver:
    """One-dimensional Yee-grid solver for waveguide co-simulation.

    The solver applies a quadratic-ramp multiplicative absorbing boundary at
    each end. It is a bounded reference implementation for pulse propagation,
    dispersion, and loss checks; use :class:`FDTD2DSolver` when split-field
    Berenger PML is required.
    """

    def __init__(
        self,
        grid_size: int = 1000,
        dx_um: float = 0.01,
        dt_factor: float = 0.5,
        refractive_index: float = 3.48,
        boundary_cells: int = 20,
    ):
        _require_count(grid_size, "grid_size", minimum=3)
        _require_count(boundary_cells, "boundary_cells", minimum=1)
        if boundary_cells > grid_size:
            raise ValueError("boundary_cells cannot exceed grid_size")
        _require_positive(dx_um, "dx_um")
        _require_positive(dt_factor, "dt_factor")
        if dt_factor > 1.0:
            raise ValueError("dt_factor must not exceed the one-dimensional CFL limit of 1")
        _require_positive(refractive_index, "refractive_index")

        self.grid_size = grid_size
        self.dx = dx_um * 1e-6
        self.c0 = 3e8
        self.n = refractive_index
        self.v = self.c0 / self.n
        self.dt = dt_factor * self.dx / self.c0
        self.ez: npt.NDArray[np.float64] = np.zeros(grid_size, dtype=np.float64)
        self.hy: npt.NDArray[np.float64] = np.zeros(grid_size, dtype=np.float64)
        self._loss_per_metre = 0.0

        self.boundary_cells = boundary_cells
        self._abc_taper: npt.NDArray[np.float64] = np.ones(grid_size, dtype=np.float64)
        for i in range(boundary_cells):
            strength = 1.0 - 0.8 * ((boundary_cells - i) / boundary_cells) ** 2
            self._abc_taper[i] = strength
            self._abc_taper[max(0, grid_size - 1 - i)] = strength

    def set_loss(self, loss_db_per_cm: float) -> None:
        """Set non-negative propagation loss in decibels per centimetre."""
        _require_non_negative(loss_db_per_cm, "loss_db_per_cm")
        self._loss_per_metre = loss_db_per_cm * 100.0

    def inject_pulse(
        self,
        position: int,
        wavelength_nm: float = 1550.0,
        amplitude: float = 1.0,
        phase: float = 0.0,
    ) -> None:
        """Inject a Gaussian-envelope optical pulse at a grid position."""
        _require_count(position, "position")
        if position >= self.grid_size:
            raise ValueError(f"position {position} is outside a {self.grid_size}-cell grid")
        _require_positive(wavelength_nm, "wavelength_nm")
        _require_non_negative(amplitude, "amplitude")
        _require_finite(phase, "phase")

        freq = self.c0 / (wavelength_nm * 1e-9)
        sigma = 20
        for i in range(max(0, position - 3 * sigma), min(self.grid_size, position + 3 * sigma)):
            r = (i - position) / sigma
            envelope = amplitude * math.exp(-0.5 * r * r)
            self.ez[i] = envelope * math.cos(2 * math.pi * freq * 0 + phase)

    def step(self, n_steps: int = 1) -> None:
        """Advance the simulation by ``n_steps`` timesteps."""
        _require_count(n_steps, "n_steps")
        coeff_e = self.dt / (self.dx * self.n**2 * 8.854e-12)
        coeff_h = self.dt / (self.dx * 4 * math.pi * 1e-7)

        if self._loss_per_metre > 0:
            alpha = self._loss_per_metre * np.log(10) / 20.0
            loss_factor = math.exp(-alpha * self.dx)
        else:
            loss_factor = 1.0

        for _ in range(n_steps):
            self.hy[:-1] += coeff_h * (self.ez[1:] - self.ez[:-1])
            self.ez[1:] += coeff_e * (self.hy[1:] - self.hy[:-1])
            if loss_factor < 1.0:
                self.ez *= loss_factor
            self.ez *= self._abc_taper
            self.hy *= self._abc_taper

    def field_energy(self) -> float:
        """Return total squared electromagnetic field energy."""
        return float(np.sum(self.ez**2) + np.sum(self.hy**2))

    def snapshot(self) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Return independent copies of the electric and magnetic fields."""
        return self.ez.copy(), self.hy.copy()


class FDTD2DSolver:
    """Two-dimensional TE Yee-grid solver with split-field Berenger PML."""

    def __init__(
        self,
        nx: int = 200,
        ny: int = 100,
        dx_um: float = 0.01,
        dy_um: float = 0.01,
        dt_factor: float = 0.5,
        pml_layers: int = 10,
    ):
        _require_count(nx, "nx", minimum=3)
        _require_count(ny, "ny", minimum=3)
        _require_count(pml_layers, "pml_layers", minimum=1)
        if pml_layers >= min(nx, ny):
            raise ValueError("pml_layers must be smaller than both grid dimensions")
        _require_positive(dx_um, "dx_um")
        _require_positive(dy_um, "dy_um")
        _require_positive(dt_factor, "dt_factor")
        if dt_factor > 1.0:
            raise ValueError("dt_factor must not exceed the two-dimensional CFL scale factor of 1")

        self.nx = nx
        self.ny = ny
        self.dx = dx_um * 1e-6
        self.dy = dy_um * 1e-6
        self.c0 = 3e8
        ds_min = min(self.dx, self.dy)
        self.dt = dt_factor * ds_min / (self.c0 * math.sqrt(2))
        self.pml_layers = pml_layers

        self.ezx: np.ndarray[Any, Any] = np.zeros((nx, ny), dtype=np.float64)
        self.ezy: np.ndarray[Any, Any] = np.zeros((nx, ny), dtype=np.float64)
        self.ez: np.ndarray[Any, Any] = np.zeros((nx, ny), dtype=np.float64)
        self.hx: np.ndarray[Any, Any] = np.zeros((nx, ny), dtype=np.float64)
        self.hy: np.ndarray[Any, Any] = np.zeros((nx, ny), dtype=np.float64)
        self.n_map: npt.NDArray[np.float64] = np.ones((nx, ny), dtype=np.float64)
        self.sigma_x: npt.NDArray[np.float64] = np.zeros((nx, ny), dtype=np.float64)
        self.sigma_y: npt.NDArray[np.float64] = np.zeros((nx, ny), dtype=np.float64)
        self._build_pml()

    def _build_pml(self) -> None:
        """Construct Berenger PML electric-conductivity profiles."""
        p = self.pml_layers
        sigma_max = 5.0 / (120.0 * math.pi * self.dx)
        for i in range(p):
            sx = sigma_max * ((p - i) / p) ** 3
            self.sigma_x[i, :] = sx
            self.sigma_x[self.nx - 1 - i, :] = sx
            self.sigma_y[:, i] = sx
            self.sigma_y[:, self.ny - 1 - i] = sx

    def set_waveguide(
        self,
        y_center: int,
        width_cells: int,
        refractive_index: float = 3.48,
        x_start: int = 0,
        x_end: Optional[int] = None,
    ) -> None:
        """Define a horizontal waveguide stripe on the material map."""
        _require_count(y_center, "y_center")
        _require_count(width_cells, "width_cells", minimum=1)
        _require_count(x_start, "x_start")
        if not 0 <= y_center < self.ny:
            raise ValueError(f"y_center {y_center} is outside a {self.ny}-cell grid")
        if x_end is not None:
            _require_count(x_end, "x_end")
        _require_positive(refractive_index, "refractive_index")
        if refractive_index < 1.0:
            raise ValueError(f"Invalid refractive index: {refractive_index}. Must be >= 1.0.")

        effective_end = self.nx if x_end is None else x_end
        effective_start = max(0, min(self.nx, x_start))
        effective_end = max(0, min(self.nx, effective_end))
        if effective_start >= effective_end:
            raise ValueError("x_start must be smaller than x_end after grid clipping")
        y_lo = max(0, min(self.ny, y_center - width_cells // 2))
        y_hi = max(0, min(self.ny, y_lo + width_cells))
        self.n_map[effective_start:effective_end, y_lo:y_hi] = refractive_index

    def inject_source(
        self,
        x: int,
        y: int,
        wavelength_nm: float = 1550.0,
        amplitude: float = 1.0,
        sigma_cells: int = 10,
    ) -> None:
        """Inject a two-dimensional Gaussian electric-field source."""
        _require_count(x, "x")
        _require_count(y, "y")
        _require_count(sigma_cells, "sigma_cells", minimum=1)
        _require_positive(wavelength_nm, "wavelength_nm")
        _require_non_negative(amplitude, "amplitude")
        if not (0 <= x < self.nx) or not (0 <= y < self.ny):
            raise ValueError(f"Source injection ({x}, {y}) out of bounds [{self.nx}, {self.ny}]")

        freq = self.c0 / (wavelength_nm * 1e-9)
        for ix in range(max(0, x - 3 * sigma_cells), min(self.nx, x + 3 * sigma_cells)):
            for iy in range(max(0, y - 3 * sigma_cells), min(self.ny, y + 3 * sigma_cells)):
                dx_r = (ix - x) / sigma_cells
                dy_r = (iy - y) / sigma_cells
                envelope = amplitude * math.exp(-0.5 * (dx_r**2 + dy_r**2))
                self.ez[ix, iy] = envelope * math.cos(2 * math.pi * freq * 0)

    def step(self, n_steps: int = 1) -> None:
        """Advance the TE simulation by ``n_steps`` timesteps."""
        _require_count(n_steps, "n_steps")
        if not np.all(np.isfinite(self.n_map)) or np.any(self.n_map <= 0):
            raise ValueError("Refractive index must be finite and > 0 in all cells.")

        eps0 = 8.854e-12
        mu0 = 4 * math.pi * 1e-7
        eps_map = eps0 * self.n_map**2
        cx_a = (eps_map - self.sigma_x * self.dt / 2.0) / (eps_map + self.sigma_x * self.dt / 2.0)
        cx_b = self.dt / ((eps_map + self.sigma_x * self.dt / 2.0) * self.dx)
        cy_a = (eps_map - self.sigma_y * self.dt / 2.0) / (eps_map + self.sigma_y * self.dt / 2.0)
        cy_b = self.dt / ((eps_map + self.sigma_y * self.dt / 2.0) * self.dy)

        smag_y = self.sigma_y * (mu0 / eps0)
        smag_x = self.sigma_x * (mu0 / eps0)
        chx_a = (mu0 - smag_y * self.dt / 2.0) / (mu0 + smag_y * self.dt / 2.0)
        chx_b = self.dt / ((mu0 + smag_y * self.dt / 2.0) * self.dy)
        chy_a = (mu0 - smag_x * self.dt / 2.0) / (mu0 + smag_x * self.dt / 2.0)
        chy_b = self.dt / ((mu0 + smag_x * self.dt / 2.0) * self.dx)

        for _ in range(n_steps):
            self.hx[:, :-1] = chx_a[:, :-1] * self.hx[:, :-1] - chx_b[:, :-1] * (
                self.ez[:, 1:] - self.ez[:, :-1]
            )
            self.hy[:-1, :] = chy_a[:-1, :] * self.hy[:-1, :] + chy_b[:-1, :] * (
                self.ez[1:, :] - self.ez[:-1, :]
            )
            self.ezx[1:, :] = cx_a[1:, :] * self.ezx[1:, :] + cx_b[1:, :] * (
                self.hy[1:, :] - self.hy[:-1, :]
            )
            self.ezy[:, 1:] = cy_a[:, 1:] * self.ezy[:, 1:] - cy_b[:, 1:] * (
                self.hx[:, 1:] - self.hx[:, :-1]
            )
            self.ez = self.ezx + self.ezy

    def field_energy(self) -> float:
        """Return total squared electromagnetic field energy."""
        return float(np.sum(self.ez**2) + np.sum(self.hx**2) + np.sum(self.hy**2))

    def field_at_point(self, x: int, y: int) -> float:
        """Return the electric field at one in-bounds grid point."""
        _require_count(x, "x")
        _require_count(y, "y")
        if x >= self.nx or y >= self.ny:
            raise ValueError(f"field point ({x}, {y}) out of bounds [{self.nx}, {self.ny}]")
        return float(self.ez[x, y])

    def cross_section(self, x: int) -> np.ndarray[Any, Any]:
        """Return an independent electric-field cross-section at ``x``."""
        _require_count(x, "x")
        if x >= self.nx:
            raise ValueError(f"cross-section index {x} out of bounds for nx={self.nx}")
        return self.ez[x, :].copy()

    def snapshot(self) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Return independent copies of all field components."""
        return self.ez.copy(), self.hx.copy(), self.hy.copy()


__all__ = ["FDTD2DSolver", "FDTDSolver"]
