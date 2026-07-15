# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wong and Wang 2006 reduced decision circuit

"""Publication-faithful reduced Wong-Wang decision circuit.

The implementation follows the two-population Appendix model in Wong and
Wang (2006), DOI ``10.1523/JNEUROSCI.3733-05.2006``.  The two NMDA gating
variables are advanced with explicit Euler and the input-current noise is an
Ornstein-Uhlenbeck process with the published AMPA time constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_A = 270.0
_B = 108.0
_D = 0.154
_DEFAULT_S = 0.1


@dataclass
class WongWangUnit:
    """Reduced two-choice decision circuit from Wong and Wang (2006).

    Parameters
    ----------
    s1, s2 : float, default=0.1
        Initial NMDA gating fractions for the two selective populations.
    noise1, noise2 : float, default=0.0
        Initial Ornstein-Uhlenbeck input-current states in nA.
    tau_s : float, default=0.1
        NMDA gating time constant in seconds.
    tau_ampa : float, default=0.002
        AMPA input-noise time constant in seconds.
    gamma : float, default=0.641
        NMDA kinetic conversion factor.
    j_n : float, default=0.2609
        Recurrent self-coupling in nA.
    j_cross : float, default=0.0497
        Cross-population inhibitory coupling magnitude in nA.
    i_0 : float, default=0.3255
        Constant background current in nA.
    sigma : float, default=0.02
        Stationary input-current noise amplitude in nA.
    dt : float, default=0.0001
        Explicit-Euler step in seconds.  The default is the 0.1 ms step stated
        in the paper; the pinned author-lab trial code uses 0.5 ms.

    Notes
    -----
    A call returns the firing rates computed from the pre-update state.  The
    caller-visible random path consumes two standard-normal samples per step.
    :meth:`step_with_gaussian_samples` exposes the same update deterministically
    for accelerator parity and source-oracle replay.
    """

    s1: float = _DEFAULT_S
    s2: float = _DEFAULT_S
    noise1: float = 0.0
    noise2: float = 0.0
    tau_s: float = 0.1
    tau_ampa: float = 0.002
    gamma: float = 0.641
    j_n: float = 0.2609
    j_cross: float = 0.0497
    i_0: float = 0.3255
    sigma: float = 0.02
    dt: float = 0.0001

    def __post_init__(self) -> None:
        """Normalise parameters and validate the initial dynamic state."""
        self._normalise_and_validate_parameters()
        self._validated_state()

    def _normalise_and_validate_parameters(self) -> None:
        for name in (
            "s1",
            "s2",
            "noise1",
            "noise2",
            "tau_s",
            "tau_ampa",
            "gamma",
            "j_n",
            "j_cross",
            "i_0",
            "sigma",
            "dt",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        for name in ("tau_s", "tau_ampa", "gamma", "dt"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("j_n", "j_cross", "sigma"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def _validated_state(self) -> tuple[float, float, float, float]:
        values = (float(self.s1), float(self.s2), float(self.noise1), float(self.noise2))
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("Wong-Wang state must remain finite")
        if not 0.0 <= values[0] <= 1.0 or not 0.0 <= values[1] <= 1.0:
            raise FloatingPointError("Wong-Wang gating state must remain in [0, 1]")
        return values

    @staticmethod
    def _phi(i_syn: float) -> float:
        i_value = float(i_syn)
        if not math.isfinite(i_value):
            raise ValueError("synaptic current must be finite")
        x = _A * i_value - _B
        scaled = -_D * x
        if scaled > 700.0:
            return 0.0
        if abs(x) < 1.0e-7:
            response = 1.0 / _D
        else:
            response = x / -math.expm1(scaled)
        if not math.isfinite(response):
            raise FloatingPointError("Wong-Wang transfer response must be finite")
        return max(0.0, response)

    @staticmethod
    def _finite_inputs(*values: float) -> tuple[float, ...]:
        converted = tuple(float(value) for value in values)
        if not all(math.isfinite(value) for value in converted):
            raise ValueError("stimuli and Gaussian samples must be finite")
        return converted

    def step_with_gaussian_samples(
        self,
        stim1: float = 0.0,
        stim2: float = 0.0,
        xi1: float = 0.0,
        xi2: float = 0.0,
    ) -> tuple[float, float]:
        """Advance one published Euler/OU step with supplied noise samples.

        Parameters
        ----------
        stim1, stim2 : float, default=0.0
            External currents for the two selective populations in nA.
        xi1, xi2 : float, default=0.0
            Independent standard-normal samples for the AMPA noise update.

        Returns
        -------
        tuple[float, float]
            Pre-update firing rates of populations one and two in Hz.

        Raises
        ------
        ValueError
            If a parameter or input is outside the numerical contract.
        FloatingPointError
            If the current state or complete candidate state is invalid.  No
            state field is changed when validation fails.
        """
        self._normalise_and_validate_parameters()
        s1, s2, noise1, noise2 = self._validated_state()
        drive1, drive2, sample1, sample2 = self._finite_inputs(stim1, stim2, xi1, xi2)

        current1 = self.j_n * s1 - self.j_cross * s2 + self.i_0 + drive1 + noise1
        current2 = self.j_n * s2 - self.j_cross * s1 + self.i_0 + drive2 + noise2
        rate1 = self._phi(current1)
        rate2 = self._phi(current2)
        ds1 = -s1 / self.tau_s + (1.0 - s1) * self.gamma * rate1
        ds2 = -s2 / self.tau_s + (1.0 - s2) * self.gamma * rate2
        noise_scale = math.sqrt(self.dt / self.tau_ampa) * self.sigma
        next_values = (
            s1 + self.dt * ds1,
            s2 + self.dt * ds2,
            noise1 - (self.dt / self.tau_ampa) * noise1 + noise_scale * sample1,
            noise2 - (self.dt / self.tau_ampa) * noise2 + noise_scale * sample2,
        )
        if not all(math.isfinite(value) for value in next_values):
            raise FloatingPointError("Wong-Wang candidate state must remain finite")
        if not 0.0 <= next_values[0] <= 1.0 or not 0.0 <= next_values[1] <= 1.0:
            raise FloatingPointError("Wong-Wang candidate gating state left [0, 1]")

        self.s1, self.s2, self.noise1, self.noise2 = next_values
        return rate1, rate2

    def step(self, stim1: float = 0.0, stim2: float = 0.0) -> tuple[float, float]:
        """Advance one stochastic Euler/OU step.

        Parameters
        ----------
        stim1, stim2 : float, default=0.0
            External currents for the two selective populations in nA.

        Returns
        -------
        tuple[float, float]
            Pre-update firing rates of populations one and two in Hz.

        Raises
        ------
        ValueError
            If a parameter, stimulus, or random sample is non-finite.
        FloatingPointError
            If the current or candidate state violates the model contract.
        """
        return self.step_with_gaussian_samples(
            stim1,
            stim2,
            float(np.random.randn()),
            float(np.random.randn()),
        )

    def simulate(
        self,
        stim1: npt.ArrayLike,
        stim2: npt.ArrayLike,
        xi: npt.ArrayLike,
        *,
        backend: str = "auto",
    ) -> dict[str, npt.NDArray[np.float64] | float]:
        """Run an atomic deterministic-sample batch on one maintained runtime.

        Parameters
        ----------
        stim1, stim2 : ArrayLike
            Per-step external currents for the two selective populations.
        xi : ArrayLike
            Interleaved standard-normal samples of length ``2 * n_steps``.
        backend : str, default="auto"
            ``python``, ``rust``, ``julia``, ``go``, ``mojo``, or ascending
            measured-latency selection.

        Returns
        -------
        dict[str, numpy.ndarray | float]
            Six post-update traces and four mutually consistent final states.

        Raises
        ------
        ValueError
            If a state, parameter, input, or backend selection is invalid.
        RuntimeError
            If an explicitly requested compiled runtime is unavailable.
        FloatingPointError
            If a backend returns an invalid candidate or malformed result.

        Notes
        -----
        This instance is unchanged unless the complete result passes the
        shared shape, finiteness, range, and trace/final consistency checks.
        """
        from sc_neurocore.accel.wong_wang import simulate_wong_wang

        result = simulate_wong_wang(
            self.s1,
            self.s2,
            self.noise1,
            self.noise2,
            self.tau_s,
            self.tau_ampa,
            self.gamma,
            self.j_n,
            self.j_cross,
            self.i_0,
            self.sigma,
            self.dt,
            stim1,
            stim2,
            xi,
            backend=backend,
        )
        self.s1 = float(result["s1_final"])
        self.s2 = float(result["s2_final"])
        self.noise1 = float(result["noise1_final"])
        self.noise2 = float(result["noise2_final"])
        return result

    def reset(self) -> None:
        """Restore the four dynamic states while preserving parameters."""
        self.s1 = _DEFAULT_S
        self.s2 = _DEFAULT_S
        self.noise1 = 0.0
        self.noise2 = 0.0
