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
    candidate_layer_profiles,
    evaluate_candidate_with_surrogate,
    optimise_candidate_policy,
)
from sc_neurocore.optimizer.sc_optimizer import LayerProfile
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


def _l_report() -> SurrogateOptimizerReport:
    return SurrogateOptimizerReport(
        config={
            "L0": _cfg(length=256, decorrelator="Sobol", precision=8, luts=1000, power=3.0),
            "L1": _cfg(length=128, decorrelator="Halton", precision=6, luts=600, power=1.5),
        },
        total_luts=1600,
        total_power_mw=4.5,
        total_latency_cycles=256,
        mean_accuracy=0.99,
        training_points=64,
        target_name="unit-fpga",
    )


class _FakeOptimiser:
    def __init__(self, report: SurrogateOptimizerReport) -> None:
        self.report = report
        self.calls: list[list[LayerProfile]] = []

    def optimise(self, network: list[LayerProfile]) -> SurrogateOptimizerReport:
        self.calls.append(network)
        return self.report


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


def test_candidate_layer_profiles_default_to_layer_neuron_counts() -> None:
    profiles = candidate_layer_profiles(_candidate())

    assert [profile.id for profile in profiles] == ["L0", "L1"]
    assert [profile.mac_count for profile in profiles] == [64, 32]
    assert profiles[0].is_critical_path is False
    assert profiles[1].is_critical_path is True


def test_candidate_layer_profiles_accept_explicit_mac_counts_and_critical_layers() -> None:
    profiles = candidate_layer_profiles(
        _candidate(),
        layer_ids=("encoder", "decoder"),
        mac_counts=(4096, 2048),
        critical_layer_indices={0},
    )

    assert profiles[0] == LayerProfile("encoder", 4096, is_critical_path=True)
    assert profiles[1] == LayerProfile("decoder", 2048, is_critical_path=False)


def test_optimise_candidate_policy_calls_surrogate_and_applies_policy() -> None:
    optimiser = _FakeOptimiser(_l_report())

    evaluation = optimise_candidate_policy(_candidate(), optimiser)

    assert optimiser.calls
    assert evaluation.policy_plan is not None
    assert evaluation.applied_policy is True
    assert evaluation.candidate.layers[0].bitstream_length == 256
    assert evaluation.candidate.layers[0].decorrelation == DecorrelationStrategy.SOBOL
    assert evaluation.candidate.layers[1].bitstream_length == 128


def test_evaluate_candidate_with_surrogate_sets_search_scores() -> None:
    optimiser = _FakeOptimiser(_l_report())

    evaluation = evaluate_candidate_with_surrogate(_candidate(), optimiser)

    assert evaluation.candidate.accuracy == 0.99
    assert evaluation.candidate.fitness == 0.99


def test_candidate_layer_profiles_reject_mismatched_mac_counts() -> None:
    with pytest.raises(ValueError, match="mac_counts length must match"):
        candidate_layer_profiles(_candidate(), mac_counts=[64])  # candidate has two layers


class _NoReportOptimiser:
    def optimise(self, network: list[LayerProfile]) -> None:
        return None


def test_optimise_candidate_policy_requires_a_report() -> None:
    with pytest.raises(RuntimeError, match="returned no report"):
        optimise_candidate_policy(_candidate(), _NoReportOptimiser())


def test_optimise_candidate_policy_returns_unapplied_for_infeasible_report() -> None:
    # A report that rejected a layer is infeasible, so the bridge returns the
    # candidate unchanged with no policy plan rather than applying a partial fit.
    infeasible = SurrogateOptimizerReport(
        config={},
        total_luts=0,
        total_power_mw=0.0,
        total_latency_cycles=0,
        mean_accuracy=0.0,
        training_points=0,
        target_name="unit-fpga",
        rejected_layers=["L0"],
    )

    evaluation = optimise_candidate_policy(_candidate(), _FakeOptimiser(infeasible))

    assert evaluation.applied_policy is False
    assert evaluation.policy_plan is None
    assert evaluation.report is infeasible
