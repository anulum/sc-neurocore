# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC project specification for the 2,560-cell WM ring

"""Frozen specification and observables for ``SC-COMPTE-WM-NETWORK``.

The network identity is SC-NeuroCore-owned.  Its control parameters are
derived from Compte et al. (2000), but the reproducible discretisation,
compact cue profile, random-stream contract, and statistical estimators are
project decisions.  Configuration alone is not evidence of a persistent bump
or distractor resistance; those claims belong to executable ensemble receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, cast, Literal

import numpy as np

PopulationKind = Literal["excitatory", "inhibitory"]
ProjectionKind = Literal["ee", "ei", "ie", "ii"]


def _positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def circular_distance_deg(
    angles_deg: np.ndarray[Any, Any] | list[float] | tuple[float, ...],
    center_deg: float,
) -> np.ndarray[Any, Any]:
    """Return shortest signed distances from *center_deg* in ``[-180, 180)``."""
    if not math.isfinite(center_deg):
        raise ValueError("center_deg must be finite")
    angles = np.asarray(angles_deg, dtype=np.float64)
    if angles.ndim != 1 or not np.all(np.isfinite(angles)):
        raise ValueError("angles_deg must be a finite one-dimensional array")
    return (angles - center_deg + 180.0) % 360.0 - 180.0


def circular_displacement_deg(before_deg: float, after_deg: float) -> float:
    """Return the signed shortest displacement from *before_deg* to *after_deg*."""
    if not math.isfinite(before_deg) or not math.isfinite(after_deg):
        raise ValueError("circular angles must be finite")
    return float((after_deg - before_deg + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True, slots=True)
class SCCompteCellSpec:
    """Intrinsic LIF parameters for one population in source units."""

    capacitance_nf: float
    leak_conductance_ns: float
    leak_reversal_mv: float = -70.0
    threshold_mv: float = -50.0
    reset_mv: float = -60.0
    refractory_ms: float = 2.0

    def __post_init__(self) -> None:
        _positive_finite("capacitance_nf", self.capacitance_nf)
        _positive_finite("leak_conductance_ns", self.leak_conductance_ns)
        _positive_finite("refractory_ms", self.refractory_ms)
        potentials = (self.leak_reversal_mv, self.threshold_mv, self.reset_mv)
        if not all(math.isfinite(value) for value in potentials):
            raise ValueError("cell potentials must be finite")
        if self.reset_mv >= self.threshold_mv:
            raise ValueError("reset_mv must be below threshold_mv")


@dataclass(frozen=True, slots=True)
class SCCompteProtocolSpec:
    """Reproducible SC protocol choices for cue, delay, and response epochs."""

    cue_duration_ms: float = 250.0
    cue_peak_pa: float = 200.0
    cue_half_width_deg: float = 18.0
    delay_duration_ms: float = 8750.0
    response_duration_ms: float = 250.0
    response_current_pa: float = 500.0
    distractor_delay_ms: float = 2500.0
    statistics_window_ms: float = 500.0

    def __post_init__(self) -> None:
        for name in (
            "cue_duration_ms",
            "cue_peak_pa",
            "cue_half_width_deg",
            "delay_duration_ms",
            "response_duration_ms",
            "response_current_pa",
            "distractor_delay_ms",
            "statistics_window_ms",
        ):
            _positive_finite(name, float(getattr(self, name)))
        if self.cue_half_width_deg > 180.0:
            raise ValueError("cue_half_width_deg must not exceed 180 degrees")


@dataclass(frozen=True, slots=True)
class SCCompteWMNetworkSpec:
    """Exact public specification of the SC 2,560-cell working-memory ring.

    The default is the paper-derived control parameter set.  ``modulated=True``
    applies the reported 20 percent NMDA and 40 percent GABAA recurrent
    conductance increases.  ``structured_ei=True`` selects the separately
    reported tuned E-to-I footprint; the default E-to-I projection is uniform.
    """

    identity: str = "SC-COMPTE-WM-NETWORK"
    specification_version: str = "sc-neurocore.sc-compte-wm-network.v1"
    n_excitatory: int = 2048
    n_inhibitory: int = 512
    dt_ms: float = 0.02
    seed: int = 42
    external_rate_hz: float = 1800.0
    external_exc_conductance_ns: float = 3.1
    external_inh_conductance_ns: float = 2.38
    recurrent_ee_conductance_ns: float = 0.381
    recurrent_ei_conductance_ns: float = 0.292
    recurrent_ie_conductance_ns: float = 1.336
    recurrent_ii_conductance_ns: float = 1.024
    ee_j_plus: float = 1.62
    ee_sigma_deg: float = 18.0
    ei_j_plus: float = 1.25
    ei_sigma_deg: float = 18.0
    structured_ei: bool = False
    modulated: bool = False
    tau_ampa_ms: float = 2.0
    tau_nmda_ms: float = 100.0
    tau_nmda_rise_ms: float = 2.0
    alpha_nmda_per_ms: float = 0.5
    tau_gabaa_ms: float = 10.0
    magnesium_mm: float = 1.0
    excitatory: SCCompteCellSpec = SCCompteCellSpec(0.5, 25.0)
    inhibitory: SCCompteCellSpec = SCCompteCellSpec(0.2, 20.0, refractory_ms=1.0)
    protocol: SCCompteProtocolSpec = SCCompteProtocolSpec()

    def __post_init__(self) -> None:
        if self.identity != "SC-COMPTE-WM-NETWORK":
            raise ValueError("identity is fixed to SC-COMPTE-WM-NETWORK")
        if self.n_excitatory != 2048 or self.n_inhibitory != 512:
            raise ValueError(
                "the v1 network size is fixed to 2048 excitatory and 512 inhibitory cells"
            )
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name in (
            "dt_ms",
            "external_rate_hz",
            "external_exc_conductance_ns",
            "external_inh_conductance_ns",
            "recurrent_ee_conductance_ns",
            "recurrent_ei_conductance_ns",
            "recurrent_ie_conductance_ns",
            "recurrent_ii_conductance_ns",
            "ee_j_plus",
            "ee_sigma_deg",
            "ei_j_plus",
            "ei_sigma_deg",
            "tau_ampa_ms",
            "tau_nmda_ms",
            "tau_nmda_rise_ms",
            "alpha_nmda_per_ms",
            "tau_gabaa_ms",
            "magnesium_mm",
        ):
            _positive_finite(name, float(getattr(self, name)))
        if self.dt_ms > min(self.tau_ampa_ms, self.tau_nmda_rise_ms):
            raise ValueError("dt_ms must not exceed the fastest enrolled synaptic time constant")
        if self.ee_j_plus <= 1.0 or self.ei_j_plus <= 1.0:
            raise ValueError("structured footprint peaks must exceed their unit mean")
        if self.ee_sigma_deg > 180.0 or self.ei_sigma_deg > 180.0:
            raise ValueError("footprint sigma must not exceed 180 degrees")

    @property
    def n_cells(self) -> int:
        """Return the fixed total population size (2,560)."""
        return self.n_excitatory + self.n_inhibitory

    def preferred_angles_deg(self, population: PopulationKind) -> np.ndarray[Any, Any]:
        """Return uniformly spaced preferred cues for one ring population."""
        if population not in ("excitatory", "inhibitory"):
            raise ValueError("population must be 'excitatory' or 'inhibitory'")
        count = self.n_excitatory if population == "excitatory" else self.n_inhibitory
        return np.arange(count, dtype=np.float64) * (360.0 / count)

    def recurrent_conductance_ns(self, projection: ProjectionKind) -> float:
        """Return the selected control or modulated recurrent conductance."""
        values = {
            "ee": self.recurrent_ee_conductance_ns,
            "ei": self.recurrent_ei_conductance_ns,
            "ie": self.recurrent_ie_conductance_ns,
            "ii": self.recurrent_ii_conductance_ns,
        }
        scale = 1.2 if projection in ("ee", "ei") and self.modulated else 1.0
        if projection in ("ie", "ii") and self.modulated:
            scale = 1.4
        return values[projection] * scale

    def connectivity_footprint(
        self,
        projection: ProjectionKind,
        source_angle_deg: float,
        target_angles_deg: np.ndarray[Any, Any] | list[float] | tuple[float, ...],
    ) -> np.ndarray[Any, Any]:
        """Return an exactly unit-mean footprint over the supplied targets.

        Control E-to-E connectivity is structured.  E-to-I is structured only
        when requested by ``structured_ei``.  Inhibitory-source projections and
        the default E-to-I projection return an all-ones footprint.
        """
        targets = np.asarray(target_angles_deg, dtype=np.float64)
        if targets.ndim != 1 or targets.size == 0 or not np.all(np.isfinite(targets)):
            raise ValueError("target_angles_deg must be a non-empty finite one-dimensional array")
        structured = projection == "ee" or (projection == "ei" and self.structured_ei)
        if not structured:
            return np.ones(targets.shape, dtype=np.float64)
        j_plus = self.ee_j_plus if projection == "ee" else self.ei_j_plus
        sigma = self.ee_sigma_deg if projection == "ee" else self.ei_sigma_deg
        distance = circular_distance_deg(targets, source_angle_deg)
        gaussian = np.exp(-0.5 * np.square(distance / sigma))
        gaussian_mean = float(np.mean(gaussian))
        denominator = 1.0 - gaussian_mean
        if denominator <= 0.0:
            raise ValueError("structured footprint requires non-degenerate target angles")
        j_minus = (1.0 - j_plus * gaussian_mean) / denominator
        if j_minus <= 0.0:
            raise ValueError("structured footprint parameters produce non-positive distal weights")
        weights = j_minus + (j_plus - j_minus) * gaussian
        # Remove binary reduction drift while retaining the requested peak shape.
        return weights / float(np.mean(weights))

    def cue_current_pa(
        self,
        center_deg: float,
        target_angles_deg: np.ndarray[Any, Any] | list[float] | tuple[float, ...],
        *,
        peak_pa: float | None = None,
    ) -> np.ndarray[Any, Any]:
        """Return the SC compact raised-cosine cue current on the ring.

        This deterministic 18-degree-support profile is a project decision; it
        must not be described as an equation copied from the source paper.
        """
        peak = self.protocol.cue_peak_pa if peak_pa is None else peak_pa
        _positive_finite("peak_pa", peak)
        distance = np.abs(circular_distance_deg(target_angles_deg, center_deg))
        half_width = self.protocol.cue_half_width_deg
        phase = np.minimum(distance / half_width, 1.0)
        current = 0.5 * peak * (1.0 + np.cos(np.pi * phase))
        current[distance >= half_width] = 0.0
        return cast(np.ndarray[Any, Any], current)


@dataclass(frozen=True, slots=True)
class SCCompteWMActivityStatistics:
    """Frozen population observables for one explicitly bounded time window."""

    excitatory_rate_hz: float
    inhibitory_rate_hz: float
    bump_angle_deg: float
    resultant_length: float
    circular_width_deg: float | None


def summarize_activity(
    spec: SCCompteWMNetworkSpec,
    excitatory_spike_counts: np.ndarray[Any, Any] | list[int],
    inhibitory_spike_counts: np.ndarray[Any, Any] | list[int],
    window_ms: float,
) -> SCCompteWMActivityStatistics:
    """Compute rates and circular bump observables from one spike-count window."""
    _positive_finite("window_ms", window_ms)
    exc = np.asarray(excitatory_spike_counts, dtype=np.float64)
    inh = np.asarray(inhibitory_spike_counts, dtype=np.float64)
    if exc.shape != (spec.n_excitatory,) or inh.shape != (spec.n_inhibitory,):
        raise ValueError("spike-count arrays must match the fixed network populations")
    if not np.all(np.isfinite(exc)) or not np.all(np.isfinite(inh)):
        raise ValueError("spike counts must be finite")
    if np.any(exc < 0.0) or np.any(inh < 0.0):
        raise ValueError("spike counts must be non-negative")
    total_exc = float(np.sum(exc))
    if total_exc <= 0.0:
        raise ValueError("bump statistics require at least one excitatory spike")
    angles_rad = np.deg2rad(spec.preferred_angles_deg("excitatory"))
    vector = np.sum(exc * np.exp(1j * angles_rad))
    angle_deg = float(np.rad2deg(np.angle(vector)) % 360.0)
    resultant = min(1.0, float(abs(vector) / total_exc))
    width = None if resultant <= 0.0 else float(np.rad2deg(math.sqrt(-2.0 * math.log(resultant))))
    seconds = window_ms / 1000.0
    return SCCompteWMActivityStatistics(
        excitatory_rate_hz=total_exc / (spec.n_excitatory * seconds),
        inhibitory_rate_hz=float(np.sum(inh)) / (spec.n_inhibitory * seconds),
        bump_angle_deg=angle_deg,
        resultant_length=resultant,
        circular_width_deg=width,
    )


__all__ = [
    "SCCompteCellSpec",
    "SCCompteProtocolSpec",
    "SCCompteWMActivityStatistics",
    "SCCompteWMNetworkSpec",
    "circular_displacement_deg",
    "circular_distance_deg",
    "summarize_activity",
]
