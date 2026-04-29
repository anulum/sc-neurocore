# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Synthesis evidence feedback loop

"""Compose synthesis evidence collection with surrogate SC optimisation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sc_neurocore.optimizer.observation_loader import observations_from_payload
from sc_neurocore.optimizer.sc_optimizer import LayerProfile
from sc_neurocore.optimizer.surrogate_sc_optimizer import (
    BenchmarkObservation,
    SurrogateOptimizerReport,
    SurrogateSCOptimizer,
    TargetHardwareProfile,
)
from sc_neurocore.optimizer.synthesis_evidence import build_payload_from_reports


@dataclass(frozen=True)
class SynthesisFeedbackResult:
    """Result of one measured-evidence optimiser feedback pass."""

    evidence_payload: dict[str, Any]
    observations: tuple[BenchmarkObservation, ...]
    report: SurrogateOptimizerReport


def optimise_from_synthesis_reports(
    *,
    network: list[LayerProfile],
    target: TargetHardwareProfile,
    design_path: str | Path,
    utilisation_path: str | Path,
    power_path: str | Path,
    accuracy_score: float,
    timing_path: str | Path | None = None,
    latency_cycles: int | None = None,
    clock_mhz: float | None = None,
    inferences_per_run: int | None = None,
) -> SynthesisFeedbackResult:
    """Parse synthesis reports and immediately rerun the SC optimiser.

    This helper is the local closed loop for the first production path:
    report files are parsed into strict evidence, evidence becomes measured
    observations, and those observations bias the surrogate optimiser for the
    supplied layer network.  It never invokes vendor tools or fabricates
    missing metrics; callers must provide reports and measured accuracy.
    """
    payload = build_payload_from_reports(
        design_path=design_path,
        utilisation_path=utilisation_path,
        power_path=power_path,
        timing_path=timing_path,
        accuracy_score=accuracy_score,
        latency_cycles=latency_cycles,
        clock_mhz=clock_mhz,
        inferences_per_run=inferences_per_run,
    )
    return optimise_from_evidence_payload(network=network, target=target, payload=payload)


def optimise_from_evidence_payload(
    *,
    network: list[LayerProfile],
    target: TargetHardwareProfile,
    payload: dict[str, Any],
) -> SynthesisFeedbackResult:
    """Rerun the SC optimiser from an in-memory evidence payload."""
    observations = tuple(observations_from_payload(payload))
    report = SurrogateSCOptimizer(target, observations=observations).optimise(network)
    if report is None:
        raise RuntimeError("surrogate optimiser returned no report")
    return SynthesisFeedbackResult(
        evidence_payload=payload,
        observations=observations,
        report=report,
    )
