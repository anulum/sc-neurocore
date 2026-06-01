# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mensi et al. 2012 — Generalized IF with escape-rate spiking

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class _XorShift64:
    state: int

    def __post_init__(self) -> None:
        if self.state == 0:
            self.state = 1

    def random(self) -> float:
        x = self.state & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 12) & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 25) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
        self.state = x & 0xFFFFFFFFFFFFFFFF
        sample = (self.state * 2685821657736338717) & 0xFFFFFFFFFFFFFFFF
        return float(sample >> 11) * (1.0 / 9007199254740992.0)


@dataclass
class GIFPopulationNeuron:
    """Generalized integrate-and-fire population neuron with escape-rate spiking.

    The deterministic subthreshold flow solves the coupled linear membrane and
    spike-history adaptation equations exactly over one fixed-current time step.
    Spiking then follows the Mensi et al. escape-rate hazard with a bounded
    Poisson interval probability.

    Reference: Mensi, S. et al. (2012). J. Neurophysiol. 107:1756-1775.
    """

    v: float = -65.0
    theta: float = -50.0
    eta: float = 0.0
    tau_m: float = 20.0
    tau_eta: float = 100.0
    delta_v: float = 2.0
    lambda_0: float = 0.001
    eta_increment: float = 5.0
    v_rest: float = -65.0
    v_reset: float = -65.0
    dt: float = 0.5
    seed: int = 42
    _rng: _XorShift64 = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = _XorShift64(self.seed)

    @staticmethod
    def _finite_values(values: tuple[float, ...]) -> bool:
        return all(math.isfinite(value) for value in values)

    def _valid_runtime(self) -> bool:
        return (
            self._finite_values(
                (
                    self.v,
                    self.theta,
                    self.eta,
                    self.tau_m,
                    self.tau_eta,
                    self.delta_v,
                    self.lambda_0,
                    self.eta_increment,
                    self.v_rest,
                    self.v_reset,
                    self.dt,
                )
            )
            and self.tau_m > 0.0
            and self.tau_eta > 0.0
            and self.delta_v > 0.0
            and self.lambda_0 >= 0.0
            and self.dt > 0.0
        )

    def _advance_subthreshold(self, current: float) -> tuple[float, float]:
        eta_decay = math.exp(-self.dt / self.tau_eta)
        membrane_decay = math.exp(-self.dt / self.tau_m)
        x0 = self.v - self.v_rest - current
        eta_new = self.eta * eta_decay
        if math.isclose(self.tau_m, self.tau_eta, rel_tol=1e-12, abs_tol=1e-12):
            x_new = membrane_decay * (x0 - (self.eta * self.dt / self.tau_m))
        else:
            coupling = self.tau_eta / (self.tau_eta - self.tau_m)
            x_new = (x0 * membrane_decay) - (self.eta * coupling * (eta_decay - membrane_decay))
        return self.v_rest + current + x_new, eta_new

    def _spike_probability(self, voltage: float) -> float:
        if self.lambda_0 == 0.0:
            return 0.0
        exponent = max(min((voltage - self.theta) / self.delta_v, 20.0), -745.0)
        hazard = self.lambda_0 * math.exp(exponent)
        return min(max(1.0 - math.exp(-hazard * self.dt), 0.0), 1.0)

    def step(self, current: float) -> int:
        if not math.isfinite(current) or not self._valid_runtime():
            return 0
        v_candidate, eta_candidate = self._advance_subthreshold(current)
        if not self._finite_values((v_candidate, eta_candidate)):
            return 0
        self.v = v_candidate
        self.eta = eta_candidate
        if self._rng.random() < self._spike_probability(self.v):
            self.v = self.v_reset
            self.eta += self.eta_increment
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self.eta = 0.0
        self._rng = _XorShift64(self.seed)
