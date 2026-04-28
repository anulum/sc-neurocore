# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bridge surrogate optimiser policies into SC-NAS

"""Apply surrogate optimiser layer policies to SC-NAS candidates."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from sc_neurocore.optimizer.sc_optimizer import LayerProfile
from sc_neurocore.optimizer.surrogate_sc_optimizer import (
    SurrogateLayerConfig,
    SurrogateOptimizerReport,
)

if TYPE_CHECKING:
    from sc_neurocore.nas.sc_nas_engine import (
        DecorrelationStrategy,
        FPGAResourceBudget,
        SCCandidate,
    )


class SupportsSurrogateOptimise(Protocol):
    """Minimal optimiser interface required by SC-NAS integration."""

    def optimise(self, network: list[LayerProfile]) -> SurrogateOptimizerReport | None:
        """Return a surrogate report for NAS layer profiles."""


@dataclass(frozen=True)
class NASPolicyLayer:
    """Per-layer compiler policy attached to an SC-NAS candidate."""

    layer_index: int
    layer_id: str
    neurons: int
    bitstream_length: int
    decorrelation: str
    mode: str
    precision_bits: int
    lfsr_polynomial: str
    luts_used: int
    power_mw: float
    latency_cycles: int
    accuracy_score: float


@dataclass(frozen=True)
class NASPolicyPlan:
    """Surrogate optimiser policy projected onto an SC-NAS candidate."""

    candidate: SCCandidate
    layers: tuple[NASPolicyLayer, ...]
    total_luts: int
    total_power_mw: float
    mean_accuracy: float
    target_name: str


@dataclass(frozen=True)
class NASPolicyEvaluation:
    """Result of calling the surrogate optimiser from inside NAS evaluation."""

    original_candidate: SCCandidate
    candidate: SCCandidate
    report: SurrogateOptimizerReport
    policy_plan: NASPolicyPlan | None
    applied_policy: bool


def candidate_layer_profiles(
    candidate: SCCandidate,
    *,
    layer_ids: list[str] | tuple[str, ...] | None = None,
    mac_counts: list[int] | tuple[int, ...] | None = None,
    critical_layer_indices: set[int] | frozenset[int] | None = None,
) -> list[LayerProfile]:
    """Convert an SC-NAS candidate into optimiser layer profiles."""
    ids = _layer_ids(candidate, layer_ids)
    if mac_counts is not None and len(mac_counts) != len(candidate.layers):
        raise ValueError("mac_counts length must match candidate layer count")
    critical = critical_layer_indices
    if critical is None:
        critical = {len(candidate.layers) - 1} if candidate.layers else set()

    profiles = []
    for index, layer in enumerate(candidate.layers):
        mac_count = int(mac_counts[index]) if mac_counts is not None else int(layer.neurons)
        profiles.append(
            LayerProfile(
                id=ids[index],
                mac_count=max(0, mac_count),
                is_critical_path=index in critical,
            )
        )
    return profiles


def optimise_candidate_policy(
    candidate: SCCandidate,
    optimiser: SupportsSurrogateOptimise,
    *,
    layer_ids: list[str] | tuple[str, ...] | None = None,
    mac_counts: list[int] | tuple[int, ...] | None = None,
    critical_layer_indices: set[int] | frozenset[int] | None = None,
    apply_policy: bool = True,
) -> NASPolicyEvaluation:
    """Run the surrogate optimiser for a candidate and optionally apply its policy."""
    ids = _layer_ids(candidate, layer_ids)
    profiles = candidate_layer_profiles(
        candidate,
        layer_ids=ids,
        mac_counts=mac_counts,
        critical_layer_indices=critical_layer_indices,
    )
    report = optimiser.optimise(profiles)
    if report is None:
        raise RuntimeError("surrogate optimiser returned no report for NAS candidate")
    if not report.feasible:
        return NASPolicyEvaluation(
            original_candidate=candidate,
            candidate=copy.deepcopy(candidate),
            report=report,
            policy_plan=None,
            applied_policy=False,
        )

    plan = build_nas_policy_plan(candidate, report, layer_ids=ids)
    updated = (
        apply_surrogate_policy(candidate, report, layer_ids=ids)
        if apply_policy
        else copy.deepcopy(candidate)
    )
    return NASPolicyEvaluation(
        original_candidate=candidate,
        candidate=updated,
        report=report,
        policy_plan=plan,
        applied_policy=apply_policy,
    )


def evaluate_candidate_with_surrogate(
    candidate: SCCandidate,
    optimiser: SupportsSurrogateOptimise,
    *,
    budget: FPGAResourceBudget | None = None,
    layer_ids: list[str] | tuple[str, ...] | None = None,
    mac_counts: list[int] | tuple[int, ...] | None = None,
    critical_layer_indices: set[int] | frozenset[int] | None = None,
) -> NASPolicyEvaluation:
    """Score a candidate through the surrogate optimiser for NAS search loops."""
    evaluation = optimise_candidate_policy(
        candidate,
        optimiser,
        layer_ids=layer_ids,
        mac_counts=mac_counts,
        critical_layer_indices=critical_layer_indices,
        apply_policy=True,
    )
    updated = evaluation.candidate
    updated.accuracy = evaluation.report.mean_accuracy
    resource_penalty = 0.0
    if not evaluation.report.feasible or (budget is not None and not updated.meets_budget(budget)):
        resource_penalty = 0.5
    updated.fitness = updated.accuracy - resource_penalty
    return evaluation


def build_nas_policy_plan(
    candidate: SCCandidate,
    report: SurrogateOptimizerReport,
    *,
    layer_ids: list[str] | tuple[str, ...] | None = None,
) -> NASPolicyPlan:
    """Project a surrogate optimiser report onto a NAS candidate."""
    ids = _layer_ids(candidate, layer_ids)
    layers = []
    for index, layer_id in enumerate(ids):
        cfg = _required_config(report, layer_id)
        nas_layer = candidate.layers[index]
        layers.append(
            NASPolicyLayer(
                layer_index=index,
                layer_id=layer_id,
                neurons=nas_layer.neurons,
                bitstream_length=cfg.bitstream_length,
                decorrelation=cfg.decorrelator,
                mode=cfg.mode,
                precision_bits=cfg.precision_bits,
                lfsr_polynomial=cfg.lfsr_polynomial,
                luts_used=cfg.luts_used,
                power_mw=cfg.power_used,
                latency_cycles=cfg.latency_cycles,
                accuracy_score=cfg.accuracy_score,
            )
        )

    return NASPolicyPlan(
        candidate=candidate,
        layers=tuple(layers),
        total_luts=report.total_luts,
        total_power_mw=report.total_power_mw,
        mean_accuracy=report.mean_accuracy,
        target_name=report.target_name,
    )


def apply_surrogate_policy(
    candidate: SCCandidate,
    report: SurrogateOptimizerReport,
    *,
    layer_ids: list[str] | tuple[str, ...] | None = None,
) -> SCCandidate:
    """Return a candidate copy with compatible bitstream/decorrelator settings."""
    ids = _layer_ids(candidate, layer_ids)
    updated = copy.deepcopy(candidate)
    for index, layer_id in enumerate(ids):
        cfg = _required_config(report, layer_id)
        updated.layers[index].bitstream_length = cfg.bitstream_length
        updated.layers[index].decorrelation = _to_nas_decorrelation(cfg.decorrelator)
    updated.evaluate_resources()
    return updated


def _layer_ids(
    candidate: SCCandidate, layer_ids: list[str] | tuple[str, ...] | None
) -> tuple[str, ...]:
    if layer_ids is None:
        return tuple(f"L{i}" for i in range(len(candidate.layers)))
    if len(layer_ids) != len(candidate.layers):
        raise ValueError("layer_ids length must match candidate layer count")
    return tuple(layer_ids)


def _required_config(report: SurrogateOptimizerReport, layer_id: str) -> SurrogateLayerConfig:
    try:
        return report.config[layer_id]
    except KeyError as exc:
        raise ValueError(f"surrogate report missing layer {layer_id!r}") from exc


def _to_nas_decorrelation(name: str) -> DecorrelationStrategy:
    from sc_neurocore.nas.sc_nas_engine import DecorrelationStrategy

    mapping = {
        "LFSR": DecorrelationStrategy.LFSR,
        "Sobol": DecorrelationStrategy.SOBOL,
        "Halton": DecorrelationStrategy.HALTON,
        "Hybrid": DecorrelationStrategy.HYBRID,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"decorrelator {name!r} is not supported by SC-NAS") from exc
