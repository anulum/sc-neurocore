# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Two-compartment LIF (TC-LIF) — Zhang et al. AAAI 2024

from __future__ import annotations

import math
from dataclasses import dataclass

#: Published hyperparameter profiles from Zhang et al. (2024), Table 5,
#: as ``(beta1, beta2, gamma, v_th)``. The paper trains ``beta1 = -sigmoid(c1)``
#: and ``beta2 = sigmoid(c2)`` per task and publishes no universal default;
#: these named profiles are the paper's per-dataset initialisations.
TC_LIF_PROFILES: dict[str, tuple[float, float, float, float]] = {
    "smnist_feedforward": (-0.5, 0.5, 0.5, 1.0),
    "smnist_recurrent": (-0.8, 0.4, 0.5, 1.0),
    "psmnist_feedforward": (-0.5, 0.5, 0.7, 1.5),
    "psmnist_recurrent": (-0.2, 0.8, 0.5, 1.8),
    "gsc_feedforward": (-0.5, 0.5, 0.6, 1.2),
    "gsc_recurrent": (-0.8, 0.8, 0.7, 1.25),
    "shd_feedforward": (-0.5, 0.5, 0.5, 1.5),
    "shd_recurrent": (-0.5, 0.5, 0.5, 1.5),
    "ssc_feedforward": (-0.5, 0.5, 0.5, 1.5),
    "ssc_recurrent": (-0.5, 0.5, 0.5, 1.5),
}


@dataclass
class TwoCompartmentLIFNeuron:
    """TC-LIF — Zhang et al. (2024) two-compartment spiking neuron.

    Discrete map (paper Eqs. 10–12, exact ordering U_D → U_S → S):

    U_D[t] = U_D[t-1] + beta1 * U_S[t-1] + I[t] - gamma * S[t-1]
    U_S[t] = U_S[t-1] + beta2 * U_D[t]          - v_th  * S[t-1]
    S[t]   = Theta(U_S[t] - v_th)

    One external input I[t] enters the dendritic compartment; both
    compartments reset softly through the delayed spike S[t-1]
    (subtraction terms). ``beta1 = -sigmoid(c1)`` and
    ``beta2 = sigmoid(c2)`` are trained per task in the paper, so
    ``beta1 in (-1, 0)`` and ``beta2 in (0, 1)``; defaults here are the
    published S-MNIST feedforward profile (Table 5), and every Table 5
    profile is exposed via :data:`TC_LIF_PROFILES` /
    :meth:`from_profile`. The right-continuous ``Theta(0) = 1``
    convention and the [0, 10] public bound on ``gamma`` are repository
    specialisations.

    Reference: Zhang, S., Yang, Q., Ma, C., Wu, J., Li, H. & Tan, K.C.
    (2024). AAAI 38(15):16838–16847. DOI 10.1609/aaai.v38i15.29625.
    """

    u_d: float = 0.0
    u_s: float = 0.0
    s_prev: float = 0.0
    beta1: float = -0.5
    beta2: float = 0.5
    gamma: float = 0.5
    v_th: float = 1.0

    def __post_init__(self) -> None:
        self._validate_configuration()

    @classmethod
    def from_profile(cls, profile: str) -> "TwoCompartmentLIFNeuron":
        """Construct the neuron from a named Table 5 profile."""
        if profile not in TC_LIF_PROFILES:
            raise ValueError(
                f"unknown TC-LIF profile {profile!r}; known profiles: {sorted(TC_LIF_PROFILES)}"
            )
        beta1, beta2, gamma, v_th = TC_LIF_PROFILES[profile]
        return cls(beta1=beta1, beta2=beta2, gamma=gamma, v_th=v_th)

    def _validate_configuration(self) -> None:
        values = (
            self.u_d,
            self.u_s,
            self.s_prev,
            self.beta1,
            self.beta2,
            self.gamma,
            self.v_th,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("TC-LIF state and parameters must be finite")
        if not -1e6 <= self.u_d <= 1e6:
            raise ValueError("u_d must be within [-1e6, 1e6]")
        if not -1e6 <= self.u_s <= 1e6:
            raise ValueError("u_s must be within [-1e6, 1e6]")
        if self.s_prev not in (0.0, 1.0):
            raise ValueError("s_prev must be 0.0 or 1.0")
        if not -1.0 < self.beta1 < 0.0:
            raise ValueError("beta1 must lie in the open interval (-1, 0)")
        if not 0.0 < self.beta2 < 1.0:
            raise ValueError("beta2 must lie in the open interval (0, 1)")
        if not 0.0 <= self.gamma <= 10.0:
            raise ValueError("gamma must be within [0, 10]")
        if not 0.0 < self.v_th <= 100.0:
            raise ValueError("v_th must be within (0, 100]")

    def step(self, i_ext: float) -> int:
        if not math.isfinite(i_ext):
            raise ValueError("i_ext must be finite")
        self._validate_configuration()

        # With a finite input, |state| <= 1e6, |beta| < 1, gamma <= 10 and
        # v_th <= 100, both candidates stay finite in binary64 (the bounded
        # terms are absorbed below one ULP of the float maximum), so no
        # separate candidate finiteness branch exists; runaway accumulation
        # is rejected by the state bound at the start of the next step.
        u_d_candidate = self.u_d + self.beta1 * self.u_s + i_ext - self.gamma * self.s_prev
        u_s_candidate = self.u_s + self.beta2 * u_d_candidate - self.v_th * self.s_prev
        spike = 1 if u_s_candidate >= self.v_th else 0

        self.u_d = u_d_candidate
        self.u_s = u_s_candidate
        self.s_prev = float(spike)
        return spike

    def reset(self) -> None:
        self.u_d = 0.0
        self.u_s = 0.0
        self.s_prev = 0.0
