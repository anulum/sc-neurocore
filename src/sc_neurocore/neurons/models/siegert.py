# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Siegert 1951 — mean-field LIF firing rate

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SiegertTransferFunction:
    """Siegert 1951 — mean-field LIF firing rate.

    Analytical stationary firing rate of a LIF neuron driven by
    Gaussian white noise: r = [tau_rp + tau_m * sqrt(pi) *
    integral(exp(u^2)*(1+erf(u)), u_reset..u_thresh)]^{-1}
    Uses Gauss-Hermite quadrature approximation.
    """

    tau_m: float = 20.0  # ms, membrane time constant
    tau_rp: float = 2.0  # ms, refractory period
    v_threshold: float = -50.0  # mV
    v_reset: float = -70.0  # mV
    v_rest: float = -65.0  # mV

    def step(self, current: float) -> float:
        """Return instantaneous firing rate (Hz) for given mean input current."""
        mu = self.v_rest + current
        sigma = max(abs(current) * 0.1, 1e-6)
        u_th = (self.v_threshold - mu) / sigma
        u_re = (self.v_reset - mu) / sigma
        # Gauss-Legendre quadrature over [u_re, u_th]
        n_quad = 40
        u_pts, w_pts = np.polynomial.legendre.leggauss(n_quad)
        half_range = 0.5 * (u_th - u_re)
        mid = 0.5 * (u_th + u_re)
        u_scaled = half_range * u_pts + mid
        integrand = np.exp(np.clip(u_scaled**2, None, 50.0)) * (1.0 + _erf_approx(u_scaled))
        integral_val = float(half_range * np.sum(w_pts * integrand))
        t_isi = self.tau_rp + self.tau_m * np.sqrt(np.pi) * integral_val
        return 1000.0 / max(t_isi, 0.01)  # Hz

    def reset(self) -> None:
        pass


def _erf_approx(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun 7.1.26 rational approximation."""
    sign = np.sign(x)
    a = np.abs(x)
    p = 0.3275911
    t = 1.0 / (1.0 + p * a)
    coeffs = np.array([0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429])
    poly = t * (coeffs[0] + t * (coeffs[1] + t * (coeffs[2] + t * (coeffs[3] + t * coeffs[4]))))
    return sign * (1.0 - poly * np.exp(-a * a))
