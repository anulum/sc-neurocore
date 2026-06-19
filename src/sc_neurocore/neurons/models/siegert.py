# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Siegert 1951 — mean-field LIF firing rate

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SiegertTransferFunction:
    """Siegert 1951 — mean-field LIF firing rate.

    Analytical stationary firing rate of a LIF neuron driven by
    Gaussian white noise: r = [tau_rp + tau_m * sqrt(pi) *
    integral(exp(u^2)*(1+erf(u)), u_reset..u_thresh)]^{-1}
    Uses Gauss-Hermite quadrature approximation.

    Reference: Siegert, A.J.F. (1951). Phys. Rev. 81:617–623.
    """

    tau_m: float = 20.0  # ms, membrane time constant
    tau_rp: float = 2.0  # ms, refractory period
    v_threshold: float = -50.0  # mV
    v_reset: float = -70.0  # mV
    v_rest: float = -65.0  # mV

    def __post_init__(self) -> None:
        for field in ("v_threshold", "v_reset", "v_rest"):
            value = getattr(self, field)
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
        for field in ("tau_m", "tau_rp"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be positive and finite")
        if self.v_threshold <= self.v_reset:
            raise ValueError("v_threshold must be greater than v_reset")

    def step(self, current: float) -> float:
        """Return instantaneous firing rate (Hz) for given mean input current."""
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()
        mu = self.v_rest + current
        if not math.isfinite(mu):
            raise ValueError("Siegert runtime mean voltage must remain finite")
        sigma = max(abs(current) * 0.1, 1e-6)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("Siegert runtime diffusion scale must remain finite and positive")
        u_th = (self.v_threshold - mu) / sigma
        u_re = (self.v_reset - mu) / sigma
        if not math.isfinite(u_th) or not math.isfinite(u_re) or u_th <= u_re:
            raise ValueError("Siegert runtime first-passage bounds must be finite and ordered")
        # Gauss-Legendre quadrature over [u_re, u_th]
        n_quad = 40
        u_pts, w_pts = np.polynomial.legendre.leggauss(n_quad)
        half_range = 0.5 * (u_th - u_re)
        mid = 0.5 * (u_th + u_re)
        if not math.isfinite(half_range) or not math.isfinite(mid) or half_range <= 0.0:
            raise ValueError("Siegert runtime quadrature interval must remain finite")
        u_scaled = half_range * u_pts + mid
        integrand = np.exp(np.clip(u_scaled**2, None, 50.0)) * (1.0 + _erf_approx(u_scaled))
        if not np.all(np.isfinite(integrand)):
            raise ValueError("Siegert runtime integrand must remain finite")
        integral_val = float(half_range * np.sum(w_pts * integrand))
        if not math.isfinite(integral_val) or integral_val < 0.0:
            raise ValueError("Siegert runtime integral must remain finite and non-negative")
        t_isi = self.tau_rp + self.tau_m * np.sqrt(np.pi) * integral_val
        if not math.isfinite(t_isi) or t_isi < self.tau_rp:
            raise ValueError("Siegert runtime inter-spike interval must remain finite")
        rate = 1000.0 / t_isi
        max_rate = 1000.0 / self.tau_rp
        if not math.isfinite(rate) or rate < 0.0 or rate > max_rate:
            raise ValueError("Siegert runtime rate must remain finite and refractory-bounded")
        return float(rate)  # Hz

    def reset(self) -> None:
        pass

    def _validate_runtime_state(self) -> None:
        try:
            self.__post_init__()
        except ValueError as exc:
            raise ValueError(f"Siegert runtime parameters invalid: {exc}") from exc


def _erf_approx(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Abramowitz & Stegun 7.1.26 rational approximation."""
    sign = np.sign(x)
    a = np.abs(x)
    p = 0.3275911
    t = 1.0 / (1.0 + p * a)
    coeffs = np.array([0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429])
    poly = t * (coeffs[0] + t * (coeffs[1] + t * (coeffs[2] + t * (coeffs[3] + t * coeffs[4]))))
    result: np.ndarray[Any, Any] = sign * (1.0 - poly * np.exp(-a * a))
    return result
