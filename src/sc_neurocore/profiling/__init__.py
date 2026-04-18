# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.profiling -- Tier: research (experimental /

"""SNN profiling: energy tracking + spike-level training diagnostics."""

__tier__ = "research"

from .energy import EnergyMetrics, track_energy
from .spike_profiler import SpikeProfiler, LayerStats, ProfileReport, Pathology

__all__ = [
    "EnergyMetrics",
    "track_energy",
    "SpikeProfiler",
    "LayerStats",
    "ProfileReport",
    "Pathology",
]
