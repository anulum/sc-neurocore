# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fault-injection resilience mode tests

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.fault_injection import (
    DegradationPlan,
    DegradationAction,
    FaultInjectionResilienceMode,
    FaultModel,
    RadiationProfile,
    ResilienceModeConfig,
    ResilienceModeReport,
    ResilienceModeTrialReport,
    SeededFaultObservation,
)
from sc_neurocore.fault_injection.resilience_policy import GracefulDegradationPolicy
from sc_neurocore.stochastic_doctor.diagnostics import AuditSeverity, BitstreamAuditReport


def test_resilience_mode_reports_seeded_probability_error_and_policy() -> None:
    bitstreams = np.array(
        [
            [0, 1, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    mode = FaultInjectionResilienceMode(
        ResilienceModeConfig(
            layer_id="layer0",
            radiation_profile=RadiationProfile("test", 0.25, "deterministic stress"),
            fault_models=(FaultModel.BIT_FLIP,),
            num_trials=16,
            seed=7,
            policy=GracefulDegradationPolicy(
                warning_affected_ratio=0.01,
                critical_affected_ratio=0.9,
            ),
        )
    )

    report = mode.run(bitstreams)
    payload = report.to_dict()
    trial = report.trial_reports[0]

    assert report.layer_id == "layer0"
    assert report.input_shape == (2, 8)
    assert report.nominal_probability == 0.5
    assert report.recommended_action == DegradationAction.EXTEND_BITSTREAM
    assert trial.expected_affected_bits == 4.0
    assert trial.observed_mean_affected_bits > 0.0
    assert 0.0 <= trial.mean_probability_error <= 1.0
    assert payload["trial_reports"][0]["degradation_plan"]["action"] == "extend_bitstream"


def test_resilience_mode_is_deterministic_for_same_seed() -> None:
    rng = np.random.default_rng(123)
    bitstreams = rng.integers(0, 2, size=(4, 32), dtype=np.uint8)
    config = ResilienceModeConfig(
        layer_id="det",
        radiation_profile=RadiationProfile("test", 0.1, "deterministic stress"),
        fault_models=(FaultModel.STUCK_AT_0, FaultModel.DROPOUT),
        num_trials=8,
        seed=99,
    )

    first = FaultInjectionResilienceMode(config).run(bitstreams)
    second = FaultInjectionResilienceMode(config).run(bitstreams)

    assert first.to_dict() == second.to_dict()


def test_resilience_mode_recommends_replay_for_correlated_streams() -> None:
    bitstreams = np.tile(np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8), (4, 1))
    mode = FaultInjectionResilienceMode(
        ResilienceModeConfig(
            layer_id="correlated",
            radiation_profile=RadiationProfile("zero", 0.0, "no injected faults"),
            fault_models=(FaultModel.BIT_FLIP,),
            num_trials=4,
            seed=11,
        )
    )

    report = mode.run(bitstreams)

    assert report.recommended_action == DegradationAction.REPLAY_WITH_SEED
    assert report.requires_replay is True
    assert report.trial_reports[0].degradation_plan.replay_seed == 11


def test_resilience_mode_rejects_invalid_inputs_and_config() -> None:
    with pytest.raises(ValueError, match="config"):
        FaultInjectionResilienceMode(config="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="num_trials"):
        ResilienceModeConfig(
            layer_id="bad",
            radiation_profile=RadiationProfile("test", 0.0),
            num_trials=0,
        )
    with pytest.raises(ValueError, match="layer_id"):
        ResilienceModeConfig(
            layer_id="",
            radiation_profile=RadiationProfile("test", 0.0),
        )
    with pytest.raises(ValueError, match="fault_models"):
        ResilienceModeConfig(
            layer_id="bad",
            radiation_profile=RadiationProfile("test", 0.0),
            fault_models=(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="seed"):
        ResilienceModeConfig(
            layer_id="bad",
            radiation_profile=RadiationProfile("test", 0.0),
            seed=True,  # type: ignore[arg-type]
        )

    mode = FaultInjectionResilienceMode(
        ResilienceModeConfig(
            layer_id="bad-shape",
            radiation_profile=RadiationProfile("test", 0.0),
        )
    )
    with pytest.raises(ValueError, match="numpy.ndarray"):
        mode.run([[0, 1]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        mode.run(np.array([[0.0, np.nan]], dtype=np.float64))
    with pytest.raises(ValueError, match="0/1"):
        mode.run(np.array([[0, 2]], dtype=np.uint8))


def test_trial_report_rejects_invalid_contracts() -> None:
    observation = SeededFaultObservation(
        layer_id="L0",
        seed=1,
        fault_model=FaultModel.BIT_FLIP,
        ber=0.1,
        affected_bits=1,
        bitstream_length=8,
        affected_ratio=0.125,
        audit=BitstreamAuditReport(
            layer="L0",
            stream_length=8,
            num_neurons=1,
            status=AuditSeverity.OK,
            max_correlation=0.0,
        ),
    )
    plan = DegradationPlan(
        action=DegradationAction.NOMINAL,
        observation=observation,
        recommended_bitstream_length=8,
        replay_seed=1,
        reason="ok",
    )
    with pytest.raises(ValueError, match="mean_probability_error"):
        ResilienceModeTrialReport(
            fault_model=FaultModel.BIT_FLIP,
            ber=0.1,
            num_trials=4,
            bit_count=8,
            expected_affected_bits=1.0,
            observed_mean_affected_bits=1.0,
            observed_std_affected_bits=0.0,
            mean_probability_error=-0.1,
            p95_probability_error=0.1,
            p99_probability_error=0.1,
            max_probability_error=0.1,
            degradation_plan=plan,
        )


def test_resilience_mode_report_rejects_invalid_contracts() -> None:
    bitstreams = np.array([[0, 1, 0, 1]], dtype=np.uint8)
    mode = FaultInjectionResilienceMode(
        ResilienceModeConfig(
            layer_id="L0",
            radiation_profile=RadiationProfile("test", 0.1),
            fault_models=(FaultModel.BIT_FLIP,),
            num_trials=2,
            seed=3,
        )
    )
    reference = mode.run(bitstreams)
    with pytest.raises(ValueError, match="nominal_probability"):
        ResilienceModeReport(
            layer_id=reference.layer_id,
            radiation_profile=reference.radiation_profile,
            seed=reference.seed,
            input_shape=reference.input_shape,
            nominal_probability=1.5,
            recommended_action=reference.recommended_action,
            trial_reports=reference.trial_reports,
        )
