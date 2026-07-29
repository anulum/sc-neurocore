# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Brunel-Wang 2001 pyramidal LIF cell

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BrunelWangNeuron:
    """Brunel-Wang pyramidal LIF cell with four aggregate synaptic gates.

    Reference: Brunel, N. & Wang, X.J. (2001). Effects of neuromodulation in a
    cortical network model of object working memory dominated by
    recurrent inhibition. J Comput Neurosci 11:63-85.

    This is the excitatory-cell specialization from Methods 2.2--2.3.  The
    four values supplied to :meth:`step` are already-summed channel gating
    variables, not spike counts and not internally integrated synapses.  One
    public step holds those gates constant and applies explicit midpoint RK2,
    matching the paper's stated second-order integration class at ``0.1 ms``.

    Notes
    -----
    The retained ``tau_*`` synaptic parameters document the source boundary
    and remain configurable metadata for network adapters.  They do not imply
    that this single-cell object owns presynaptic channel states.
    """

    v: float = -70.0
    v_rest: float = -70.0
    v_reset: float = -55.0
    v_threshold: float = -50.0
    tau_m: float = 20.0
    tau_ref: float = 2.0
    tau_ampa: float = 2.0
    tau_nmda_rise: float = 2.0
    tau_nmda_decay: float = 100.0
    tau_gaba: float = 10.0
    g_ampa_ext: float = 2.08
    g_ampa_rec: float = 0.104
    g_nmda: float = 0.327
    g_gaba: float = 1.25
    v_ampa: float = 0.0
    v_nmda: float = 0.0
    v_gaba: float = -70.0
    C_m: float = 0.5
    mg_conc: float = 1.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "v",
            "v_rest",
            "v_reset",
            "v_threshold",
            "tau_m",
            "tau_ref",
            "tau_ampa",
            "tau_nmda_rise",
            "tau_nmda_decay",
            "tau_gaba",
            "g_ampa_ext",
            "g_ampa_rec",
            "g_nmda",
            "g_gaba",
            "v_ampa",
            "v_nmda",
            "v_gaba",
            "C_m",
            "mg_conc",
            "dt",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        for name in (
            "tau_m",
            "tau_ref",
            "tau_ampa",
            "tau_nmda_rise",
            "tau_nmda_decay",
            "tau_gaba",
            "C_m",
            "dt",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("g_ampa_ext", "g_ampa_rec", "g_nmda", "g_gaba", "mg_conc"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self._validate_voltage(self.v)
        self._ref_remaining = 0.0

    @staticmethod
    def _validate_voltage(v: float) -> float:
        voltage = float(v)
        if not math.isfinite(voltage):
            raise FloatingPointError("Brunel-Wang voltage state must be finite")
        return voltage

    @staticmethod
    def _validate_nonnegative(name: str, value: float) -> float:
        scalar = float(value)
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return scalar

    @staticmethod
    def _validate_aggregate_gate(name: str, value: float) -> float:
        gate = float(value)
        if not math.isfinite(gate) or gate < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return gate

    def _nmda_voltage_dep(self, v: float) -> float:
        """Mg2+ block factor: 1 / (1 + [Mg2+]/3.57 * exp(-0.062 * V))."""
        voltage = self._validate_voltage(v)
        exponent = -0.062 * voltage
        if exponent > 700.0:
            return 0.0
        factor = 1.0 / (1.0 + self.mg_conc / 3.57 * math.exp(exponent))
        if not math.isfinite(factor) or factor < 0.0 or factor > 1.0:
            raise FloatingPointError("Brunel-Wang NMDA voltage factor is invalid")
        return factor

    def step(
        self,
        i_ampa_ext: float = 0.0,
        s_ampa_rec: float = 0.0,
        s_nmda_rec: float = 0.0,
        s_gaba: float = 0.0,
    ) -> int:
        """Advance one source-level midpoint-RK2 timestep atomically.

        Parameters
        ----------
        i_ampa_ext : legacy name for the summed external AMPA gate
        s_ampa_rec : summed recurrent AMPA gate
        s_nmda_rec : weighted summed recurrent NMDA gate
        s_gaba : summed recurrent GABA gate

        Returns
        -------
        int
            One when the sampled candidate reaches threshold, otherwise zero.

        Raises
        ------
        ValueError
            If configuration or an input gate is invalid.  Failure leaves the
            voltage and refractory state unchanged.
        FloatingPointError
            If either RK2 stage is non-finite.  Failure is atomic.
        """
        self.__post_init_runtime()
        ampa_ext = self._validate_aggregate_gate("i_ampa_ext", i_ampa_ext)
        ampa_rec = self._validate_aggregate_gate("s_ampa_rec", s_ampa_rec)
        nmda_rec = self._validate_aggregate_gate("s_nmda_rec", s_nmda_rec)
        gaba = self._validate_aggregate_gate("s_gaba", s_gaba)
        v = self._validate_voltage(self.v)
        ref_remaining = self._validate_nonnegative("_ref_remaining", self._ref_remaining)

        if ref_remaining > 0:
            self.v = self.v_reset
            self._ref_remaining = max(0.0, ref_remaining - self.dt)
            return 0

        k1 = self._dv_dt(v, ampa_ext, ampa_rec, nmda_rec, gaba)
        midpoint = v + 0.5 * self.dt * k1
        if not math.isfinite(midpoint):
            raise FloatingPointError("Brunel-Wang RK2 midpoint became non-finite")
        k2 = self._dv_dt(midpoint, ampa_ext, ampa_rec, nmda_rec, gaba)
        next_v = v + self.dt * k2
        if not math.isfinite(next_v):
            raise FloatingPointError("Brunel-Wang RK2 candidate became non-finite")

        self.v = next_v

        if next_v >= self.v_threshold:
            self.v = self.v_reset
            self._ref_remaining = self.tau_ref
            return 1
        return 0

    def _dv_dt(
        self,
        voltage: float,
        ampa_ext: float,
        ampa_rec: float,
        nmda_rec: float,
        gaba: float,
    ) -> float:
        """Return the source membrane derivative for fixed aggregate gates."""
        i_ampa = -self.g_ampa_ext * (voltage - self.v_ampa) * ampa_ext
        i_ampa -= self.g_ampa_rec * (voltage - self.v_ampa) * ampa_rec
        i_nmda = -self.g_nmda * self._nmda_voltage_dep(voltage) * (voltage - self.v_nmda) * nmda_rec
        i_gaba = -self.g_gaba * (voltage - self.v_gaba) * gaba
        derivative = -(voltage - self.v_rest) / self.tau_m
        derivative += (i_ampa + i_nmda + i_gaba) / self.C_m
        if not math.isfinite(derivative):
            raise FloatingPointError("Brunel-Wang membrane derivative became non-finite")
        return derivative

    def __post_init_runtime(self) -> None:
        """Validate mutated public parameters without resetting dynamic state."""
        # Dynamic-state failures retain their public boundary exception types;
        # parameter revalidation below must not obscure a corrupted voltage.
        self._validate_voltage(self.v)
        self._validate_nonnegative("_ref_remaining", self._ref_remaining)
        dynamic = (self.v, self._ref_remaining)
        try:
            self.__post_init__()
        except ValueError as exc:
            self.v, self._ref_remaining = dynamic
            raise ValueError(f"Brunel-Wang runtime parameters invalid: {exc}") from exc
        self.v, self._ref_remaining = dynamic

    def simulate(
        self,
        i_ampa_ext: object,
        s_ampa_rec: object,
        s_nmda_rec: object,
        s_gaba: object,
        *,
        backend: str = "auto",
    ) -> dict[str, object]:
        """Run a complete four-gate batch through one maintained backend."""
        from sc_neurocore.accel.brunel_wang import simulate_brunel_wang

        result = simulate_brunel_wang(
            self.v,
            self._ref_remaining,
            self.v_rest,
            self.v_reset,
            self.v_threshold,
            self.tau_m,
            self.tau_ref,
            self.g_ampa_ext,
            self.g_ampa_rec,
            self.g_nmda,
            self.g_gaba,
            self.v_ampa,
            self.v_nmda,
            self.v_gaba,
            self.C_m,
            self.mg_conc,
            self.dt,
            i_ampa_ext,
            s_ampa_rec,
            s_nmda_rec,
            s_gaba,
            backend=backend,
        )
        self.v = float(result["v_final"])
        self._ref_remaining = float(result["ref_final"])
        return result

    def reset(self) -> None:
        """Reset dynamic state while preserving every configuration field."""
        self.v = self.v_rest
        self._ref_remaining = 0.0

    def get_state(self) -> dict[str, float]:
        """Return the complete dynamic membrane/refractory state."""
        return {"v": self.v, "ref_remaining": self._ref_remaining}


__all__ = ["BrunelWangNeuron"]
