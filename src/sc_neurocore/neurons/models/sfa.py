# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Benda & Herz 2003 — Spike Frequency Adaptation IF

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class SFANeuron:
    """Benda-Herz spike-frequency adaptation IF with RK4 candidates.

    Reference: Benda, J. & Herz, A.V.M. (2003). Neural Comput. 15:2523–2564.

    The membrane and adaptation conductance state is advanced as a coupled
    ``(v, g_sfa)`` RK4 candidate. The candidate is committed only after finite
    and envelope checks pass. A spike resets voltage and adds ``delta_g`` to
    the RK4 adaptation candidate.
    """

    v: float = -70.0
    g_sfa: float = 0.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 10.0
    tau_sfa: float = 200.0
    delta_g: float = 0.5
    e_k: float = -80.0
    resistance: float = 1.0
    dt: float = 1.0

    _V_MIN: float = -200.0
    _V_MAX: float = 100.0
    _G_MAX: float = 1.0e9

    def __post_init__(self) -> None:
        self._validated_state()

    @staticmethod
    def _finite(value: float, name: str) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    @classmethod
    def _positive(cls, value: float, name: str) -> float:
        value = cls._finite(value, name)
        if value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    @classmethod
    def _nonnegative(cls, value: float, name: str) -> float:
        value = cls._finite(value, name)
        if value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return value

    def _validated_state(self) -> tuple[float, float]:
        for field in ("v", "v_rest", "v_reset", "v_threshold", "e_k"):
            self._finite(getattr(self, field), field)
        if not self._V_MIN <= self.v <= self._V_MAX:
            raise ValueError("v outside SFA safety envelope")
        if not self._V_MIN <= self.v_reset <= self._V_MAX:
            raise ValueError("v_reset outside SFA safety envelope")
        g_sfa = self._nonnegative(self.g_sfa, "g_sfa")
        if g_sfa > self._G_MAX:
            raise ValueError("g_sfa outside SFA safety envelope")
        for field in ("tau_m", "tau_sfa", "resistance", "dt"):
            self._positive(getattr(self, field), field)
        delta_g = self._nonnegative(self.delta_g, "delta_g")
        if delta_g > self._G_MAX:
            raise ValueError("delta_g outside SFA safety envelope")
        return self.v, g_sfa

    def step(self, current: float) -> int:
        """Advance one candidate-first RK4 timestep.

        Parameters
        ----------
        current:
            External drive current.

        Returns
        -------
        int
            ``1`` when the RK4 voltage candidate reaches threshold, otherwise
            ``0``. Invalid runtime inputs raise ``ValueError`` before mutation.
        """
        current = self._finite(current, "current")
        v, g_sfa = self._validated_state()

        v_candidate, g_candidate = self._rk4_candidate(v, g_sfa, current)
        for value, name in (
            (v_candidate, "voltage candidate"),
            (g_candidate, "adaptation conductance candidate"),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not self._V_MIN <= v_candidate <= self._V_MAX:
            raise ValueError("voltage candidate outside SFA safety envelope")
        if g_candidate < 0.0 or g_candidate > self._G_MAX:
            raise ValueError("adaptation conductance candidate outside SFA safety envelope")

        if v_candidate >= self.v_threshold:
            g_after_spike = g_candidate + self.delta_g
            if not math.isfinite(g_after_spike) or g_after_spike > self._G_MAX:
                raise ValueError("post-spike adaptation candidate outside SFA safety envelope")
            self.v = self.v_reset
            self.g_sfa = g_after_spike
            return 1

        self.v = v_candidate
        self.g_sfa = g_candidate
        return 0

    def reset(self) -> None:
        """Restore voltage to rest and clear adaptation conductance."""
        self.v = self.v_rest
        self.g_sfa = 0.0

    def _derivatives(self, v: float, g_sfa: float, current: float) -> tuple[float, float]:
        dv = (-(v - self.v_rest) - g_sfa * (v - self.e_k) + self.resistance * current) / self.tau_m
        dg = -g_sfa / self.tau_sfa
        return dv, dg

    def _rk4_candidate(self, v: float, g_sfa: float, current: float) -> tuple[float, float]:
        """Return the coupled RK4 candidate for ``(v, g_sfa)``."""
        k1v, k1g = self._derivatives(v, g_sfa, current)
        k2v, k2g = self._derivatives(
            v + 0.5 * self.dt * k1v,
            g_sfa + 0.5 * self.dt * k1g,
            current,
        )
        k3v, k3g = self._derivatives(
            v + 0.5 * self.dt * k2v,
            g_sfa + 0.5 * self.dt * k2g,
            current,
        )
        k4v, k4g = self._derivatives(
            v + self.dt * k3v,
            g_sfa + self.dt * k3g,
            current,
        )
        return (
            v + (self.dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
            g_sfa + (self.dt / 6.0) * (k1g + 2.0 * k2g + 2.0 * k3g + k4g),
        )
