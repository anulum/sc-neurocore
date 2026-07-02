# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hybrid Fisher-Posner LIF neuron

"""Leaky Integrate-and-Fire neuron with quantum-metabolic coupling.

This model extends the standard LIF neuron with a metabolic layer
driven by a :class:`SpinPoolMPS`.  The quantum spin pool modulates ATP
regeneration rate, introducing a non-local coupling dimension: a spike
in neuron A can alter ATP availability in distant neuron B through
shared entanglement.

Hybrid Hamiltonian
------------------

The effective dynamics combine classical membrane potential with a
quantum-metabolic current:

    dV/dt = (-(V - V_rest) + I_in + I_pump) / τ_m

where the pump current depends on quantum singlet efficiency η:

    I_pump = (η - 0.5) · 2 · ATP_level

    η = Tr(ρ_site,site+1 · P_singlet)

Spiking requires both voltage threshold crossing AND sufficient ATP:

    spike iff V ≥ V_th  AND  ATP ≥ ATP_consumption

If ATP is depleted, the neuron experiences *metabolic failure* — it
cannot fire despite suprathreshold voltage.  This models the
bioenergetic constraint on neural computation.

References
----------
- Fisher, M. P. A. "Quantum cognition: The possibility of processing
  with nuclear spins in the brain." *Annals of Physics* 362 (2015).
- Gerstner, W. & Kistler, W. M. *Spiking Neuron Models.* Cambridge
  University Press (2002).
"""

from __future__ import annotations

import math
from typing import Any

from .spin_pool import SpinPoolMPS


def _finite_float(value: float, name: str) -> float:
    """Return a finite floating-point model parameter."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value: float, name: str) -> float:
    """Return a finite, strictly positive floating-point model parameter."""
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def _nonnegative_float(value: float, name: str) -> float:
    """Return a finite, non-negative floating-point model parameter."""
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return result


def _unit_interval_float(value: float, name: str) -> float:
    """Return a finite floating-point value in the closed unit interval."""
    result = _finite_float(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _positive_unit_float(value: float, name: str) -> float:
    """Return a finite floating-point value in the interval ``(0, 1]``."""
    result = _finite_float(value, name)
    if not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return result


def _validate_neuron_id(neuron_id: int, n_sites: int) -> int:
    """Return a concrete spin-pool site index for a public neuron id."""
    if not isinstance(neuron_id, int) or isinstance(neuron_id, bool):
        raise TypeError("neuron_id must be an integer spin-pool site index")
    if neuron_id < 0:
        raise ValueError(f"neuron_id must be >= 0, got {neuron_id}")
    if neuron_id >= n_sites:
        raise ValueError(f"neuron_id {neuron_id} exceeds spin_pool.n_sites ({n_sites})")
    return neuron_id


class HybridFisherPosnerLIF:
    """LIF neuron with quantum-metabolic coupling via spin pool.

    Parameters
    ----------
    neuron_id : int
        Site index in the spin pool (determines entanglement location).
    spin_pool : SpinPoolMPS
        Shared spin pool providing non-local ATP modulation.
    dt : float
        Integration timestep in ms.
    v_rest : float
        Resting membrane potential in mV.
    v_threshold : float
        Spike threshold in mV.
    v_reset : float
        Post-spike reset potential in mV.
    tau_m : float
        Membrane time constant in ms.
    atp_initial : float
        Initial ATP level (normalised, 0–1).
    atp_consumption : float
        ATP consumed per spike.
    atp_basal_regeneration : float
        First-order ATP recovery rate independent of Posner singlet yield.
        Posner efficiency modulates this baseline rather than replacing
        cellular metabolism.
    """

    def __init__(
        self,
        neuron_id: int,
        spin_pool: SpinPoolMPS,
        dt: float = 1.0,
        v_rest: float = -70.0,
        v_threshold: float = -50.0,
        v_reset: float = -70.0,
        tau_m: float = 20.0,
        atp_initial: float = 1.0,
        atp_consumption: float = 0.05,
        atp_basal_regeneration: float = 0.001,
    ) -> None:
        if not isinstance(spin_pool, SpinPoolMPS):
            raise TypeError(f"spin_pool must be SpinPoolMPS, got {type(spin_pool).__name__}")
        neuron_id = _validate_neuron_id(neuron_id, spin_pool.n_sites)

        self.id = neuron_id
        self.spin_pool = spin_pool
        self.dt = _positive_float(dt, "dt")

        # Classical membrane parameters
        self.v_rest = _finite_float(v_rest, "v_rest")
        self.v_threshold = _finite_float(v_threshold, "v_threshold")
        self.v_reset = _finite_float(v_reset, "v_reset")
        self.tau_m = _positive_float(tau_m, "tau_m")
        self.Vm = self.v_rest

        # Alias for Population compatibility (Population reads neuron.v)

        # Metabolic parameters
        self.atp_level = _unit_interval_float(atp_initial, "atp_initial")
        self.atp_consumption = _positive_unit_float(atp_consumption, "atp_consumption")
        self.atp_basal_regeneration = _nonnegative_float(
            atp_basal_regeneration,
            "atp_basal_regeneration",
        )
        self.is_spiking = False

        # Counters for telemetry
        self._total_spikes = 0
        self._metabolic_failures = 0
        self._total_steps = 0

    def step(self, I_in: float) -> tuple[float, bool]:
        """Advance the neuron by one timestep.

        Parameters
        ----------
        I_in : float
            External input current (arbitrary units).

        Returns
        -------
        tuple[float, bool]
            ``(Vm, is_spiking)`` — membrane voltage and spike flag.
        """
        current = _finite_float(I_in, "I_in")

        # 1. Quantum-modulated ATP regeneration
        eff = self.spin_pool.get_local_atp_efficiency(self.id)
        self._total_steps += 1
        r_atp = self.atp_basal_regeneration + eff * 0.01
        self.atp_level = min(1.0, self.atp_level + r_atp * (1.0 - self.atp_level))

        # 2. Metabolic pump current: quantum-classical coupling
        #    Positive when entanglement efficiency > 0.5 (enhanced state),
        #    negative when below (depleted state).
        i_pump = (eff - 0.5) * 2.0 * self.atp_level

        # 3. Classical LIF integration with metabolic pump
        dv = (-(self.Vm - self.v_rest) + current + i_pump) / self.tau_m * self.dt
        self.Vm += dv

        # 4. Spiking logic with metabolic gate
        self.is_spiking = False
        if self.Vm >= self.v_threshold:
            if self.atp_level >= self.atp_consumption:
                # Successful spike
                self.is_spiking = True
                self.Vm = self.v_reset
                self.atp_level -= self.atp_consumption
                self._total_spikes += 1
                # Feedback into quantum layer (measurement/collapse)
                self.spin_pool.apply_measurement(self.id, 1.0)
            else:
                # Metabolic failure: insufficient ATP for spike
                self.Vm = self.v_threshold - 1.0
                self._metabolic_failures += 1

        return self.Vm, self.is_spiking

    def get_state(self) -> dict[str, Any]:
        """Return full neuron state for checkpointing."""
        return {
            "neuron_id": self.id,
            "Vm": self.Vm,
            "atp_level": self.atp_level,
            "is_spiking": self.is_spiking,
            "total_spikes": self._total_spikes,
            "metabolic_failures": self._metabolic_failures,
            "total_steps": self._total_steps,
            "v_threshold": self.v_threshold,
            "v_reset": self.v_reset,
            "tau_m": self.tau_m,
            "atp_basal_regeneration": self.atp_basal_regeneration,
        }

    def reset_state(self) -> None:
        """Reset to resting potential and full ATP."""
        self.Vm = self.v_rest
        self.atp_level = 1.0
        self.is_spiking = False
        self._total_spikes = 0
        self._metabolic_failures = 0
        self._total_steps = 0

    def reset(self) -> None:
        """Alias for reset_state (NeuronProtocol compatibility)."""
        self.reset_state()

    @property
    def v(self) -> float:
        """Membrane voltage alias for Population compatibility."""
        return self.Vm

    @v.setter
    def v(self, value: float) -> None:
        """Assign a finite membrane voltage through the Population alias."""
        self.Vm = _finite_float(value, "v")

    def __repr__(self) -> str:
        """Return compact debug telemetry for the coupled neuron."""
        return (
            f"HybridFisherPosnerLIF(id={self.id}, Vm={self.Vm:.2f}, "
            f"ATP={self.atp_level:.3f}, spikes={self._total_spikes})"
        )


class HybridFisherPosnerLIFNeuron:
    """Population-compatible wrapper for HybridFisherPosnerLIF.

    This class creates its own SpinPoolMPS and wraps ``step()`` to return
    ``0`` or ``1`` (integer spike flag) instead of ``(Vm, bool)``, making it
    compatible with ``Population(model='HybridFisherPosnerLIFNeuron', n=N)``.

    The underlying SpinPoolMPS is shared among all neurons in the same
    Population via class-level pool management.
    """

    # Class-level shared pool registry: pool_id → SpinPoolMPS
    _shared_pools: dict[int, SpinPoolMPS] = {}
    _pool_counter: int = 0

    def __init__(
        self,
        dt: float = 1.0,
        v_rest: float = -70.0,
        v_threshold: float = -50.0,
        v_reset: float = -70.0,
        tau_m: float = 20.0,
        atp_initial: float = 1.0,
        atp_consumption: float = 0.05,
        atp_basal_regeneration: float = 0.001,
        n_sites: int = 32,
    ) -> None:
        # Allocate a pool site for this neuron
        pool_id = HybridFisherPosnerLIFNeuron._pool_counter
        if pool_id not in HybridFisherPosnerLIFNeuron._shared_pools:
            HybridFisherPosnerLIFNeuron._shared_pools[pool_id] = SpinPoolMPS(n_sites=n_sites)
        pool = HybridFisherPosnerLIFNeuron._shared_pools[pool_id]

        # Find next free site in the pool
        site_idx = getattr(pool, "_next_site", 0)
        if site_idx >= pool.n_sites:
            site_idx = site_idx % pool.n_sites  # wrap around
        pool._next_site = site_idx + 1  # type: ignore[attr-defined]

        self._inner = HybridFisherPosnerLIF(
            neuron_id=site_idx,
            spin_pool=pool,
            dt=dt,
            v_rest=v_rest,
            v_threshold=v_threshold,
            v_reset=v_reset,
            tau_m=tau_m,
            atp_initial=atp_initial,
            atp_consumption=atp_consumption,
            atp_basal_regeneration=atp_basal_regeneration,
        )

    def step(self, I_in: float) -> int:
        """Advance one timestep, return 1 if spiked else 0."""
        _, spiked = self._inner.step(I_in)
        return 1 if spiked else 0

    @property
    def v(self) -> float:
        """Return the inner neuron's membrane voltage."""
        return self._inner.Vm

    @v.setter
    def v(self, value: float) -> None:
        """Assign a finite membrane voltage through the wrapper alias."""
        self._inner.v = value

    @property
    def v_threshold(self) -> float:
        """Return the inner neuron's spike threshold."""
        return self._inner.v_threshold

    @property
    def v_rest(self) -> float:
        """Return the inner neuron's resting membrane potential."""
        return self._inner.v_rest

    def get_state(self) -> dict[str, Any]:
        """Return the inner neuron's checkpointable state."""
        return self._inner.get_state()

    def reset(self) -> None:
        """Reset the wrapped neuron state for NeuronProtocol compatibility."""
        self._inner.reset_state()

    def reset_state(self) -> None:
        """Reset the wrapped neuron state to rest and full ATP."""
        self._inner.reset_state()

    def __repr__(self) -> str:
        """Return compact debug telemetry for the population wrapper."""
        return f"HybridFisherPosnerLIFNeuron({self._inner!r})"

    @classmethod
    def _reset_pools(cls) -> None:
        """Reset shared pool registry (for testing)."""
        cls._shared_pools.clear()
        cls._pool_counter = 0


__all__ = ["HybridFisherPosnerLIF", "HybridFisherPosnerLIFNeuron"]
