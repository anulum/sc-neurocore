# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Surrogate optimiser to SC-NAS bridge tests

from __future__ import annotations

import pytest

from sc_neurocore.nas.sc_nas_engine import (
    DecorrelationStrategy,
    LayerConfig,
    NeuronType,
    SCCandidate,
)
from sc_neurocore.nas.surrogate_bridge import (
    apply_surrogate_policy,
    build_nas_policy_plan,
)
from sc_neurocore.optimizer.surrogate_sc_optimizer import (
    SurrogateLayerConfig,
    SurrogateOptimizerReport,
)


def _candidate() -> SCCandidate:
    return SCCandidate(
        layers=[
            LayerConfig(64, NeuronType.LIF, 64, DecorrelationStrategy.LFSR),
            LayerConfig(32, NeuronType.ADEX, 64, DecorrelationStrategy.LFSR),
        ]
    )


def _cfg(
    *,
    length: int,
    decorrelator: str,
    precision: int,
    luts: int,
    power: float,
) -> SurrogateLayerConfig:
    return SurrogateLayerConfig(
        bitstream_length=length,
        decorrelator=decorrelator,
        mode="SC",
        precision_bits=precision,
        lfsr_polynomial="x16+x14+x13+x11+1",
        luts_used=luts,
        power_used=power,
        latency_cycles=length,
        accuracy_score=0.99,
        utility_score=0.95,
    )


def _report() -> SurrogateOptimizerReport:
    return SurrogateOptimizerReport(
        config={
            "encoder": _cfg(length=256, decorrelator="Sobol", precision=8, luts=1000, power=3.0),
            "decoder": _cfg(length=128, decorrelator="Halton", precision=6, luts=600, power=1.5),
        },
        total_luts=1600,
        total_power_mw=4.5,
        total_latency_cycles=256,
        mean_accuracy=0.99,
        training_points=64,
        target_name="unit-fpga",
    )


def test_build_nas_policy_plan_projects_report() -> None:
    plan = build_nas_policy_plan(_candidate(), _report(), layer_ids=("encoder", "decoder"))

    assert plan.target_name == "unit-fpga"
    assert plan.total_luts == 1600
    assert len(plan.layers) == 2
    assert plan.layers[0].layer_index == 0
    assert plan.layers[0].layer_id == "encoder"
    assert plan.layers[0].neurons == 64
    assert plan.layers[0].bitstream_length == 256
    assert plan.layers[0].decorrelation == "Sobol"
    assert plan.layers[0].precision_bits == 8


def test_apply_surrogate_policy_returns_updated_candidate_copy() -> None:
    original = _candidate()
    updated = apply_surrogate_policy(original, _report(), layer_ids=("encoder", "decoder"))

    assert updated is not original
    assert original.layers[0].bitstream_length == 64
    assert updated.layers[0].bitstream_length == 256
    assert updated.layers[0].decorrelation == DecorrelationStrategy.SOBOL
    assert updated.layers[1].bitstream_length == 128
    assert updated.layers[1].decorrelation == DecorrelationStrategy.HALTON
    assert updated.total_luts > 0


def test_default_layer_ids_match_l0_l1() -> None:
    report = SurrogateOptimizerReport(
        config={
            "L0": _cfg(length=512, decorrelator="LFSR", precision=8, luts=100, power=1.0),
            "L1": _cfg(length=256, decorrelator="Sobol", precision=8, luts=100, power=1.0),
        },
        total_luts=200,
        total_power_mw=2.0,
        total_latency_cycles=512,
        mean_accuracy=0.98,
        training_points=10,
        target_name="unit",
    )

    updated = apply_surrogate_policy(_candidate(), report)

    assert updated.layers[0].bitstream_length == 512
    assert updated.layers[1].decorrelation == DecorrelationStrategy.SOBOL


def test_rejects_layer_id_count_mismatch() -> None:
    with pytest.raises(ValueError, match="layer_ids length"):
        build_nas_policy_plan(_candidate(), _report(), layer_ids=("only_one",))


def test_rejects_missing_report_layer() -> None:
    with pytest.raises(ValueError, match="missing layer"):
        build_nas_policy_plan(_candidate(), _report(), layer_ids=("encoder", "missing"))


def test_rejects_unsupported_decorrelator_when_mutating_candidate() -> None:
    report = SurrogateOptimizerReport(
        config={
            "L0": _cfg(
                length=128, decorrelator="SCC_Decorrelator", precision=8, luts=100, power=1.0
            ),
            "L1": _cfg(length=128, decorrelator="LFSR", precision=8, luts=100, power=1.0),
        },
        total_luts=200,
        total_power_mw=2.0,
        total_latency_cycles=128,
        mean_accuracy=0.98,
        training_points=10,
        target_name="unit",
    )

    with pytest.raises(ValueError, match="not supported"):
        apply_surrogate_policy(_candidate(), report)
