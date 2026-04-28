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

from sc_neurocore.nas.sc_nas_engine import DecorrelationStrategy, SCCandidate
from sc_neurocore.optimizer.surrogate_sc_optimizer import (
    SurrogateLayerConfig,
    SurrogateOptimizerReport,
)


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
