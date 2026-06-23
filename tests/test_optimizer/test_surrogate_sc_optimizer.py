# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Surrogate SC optimiser tests

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from sc_neurocore.optimizer.sc_optimizer import HardwareBudget, LayerProfile
from sc_neurocore.optimizer.surrogate_sc_optimizer import (
    BenchmarkObservation,
    SurrogateSCOptimizer,
    TargetHardwareProfile,
    _RidgeSurrogate,
)


def _target(max_luts: int = 200_000, max_power_mw: float = 800.0) -> TargetHardwareProfile:
    return TargetHardwareProfile(
        name="unit-fpga",
        budget=HardwareBudget(
            max_luts=max_luts,
            max_power_mw=max_power_mw,
            max_latency_cycles=512,
        ),
    )


def _network() -> list[LayerProfile]:
    return [
        LayerProfile(id="encoder", mac_count=256, is_critical_path=True),
        LayerProfile(id="decoder", mac_count=192, is_critical_path=False),
    ]


def test_surrogate_optimizer_returns_budgeted_per_layer_config() -> None:
    opt = SurrogateSCOptimizer(_target())
    report = opt.optimise(_network())

    assert report is not None
    assert report.feasible
    assert set(report.config) == {"encoder", "decoder"}
    assert report.total_luts <= opt.target.budget.max_luts
    assert report.total_power_mw <= opt.target.budget.max_power_mw
    assert report.total_latency_cycles <= opt.target.budget.max_latency_cycles
    assert report.training_points > 0

    for cfg in report.config.values():
        assert cfg.bitstream_length in {1, 64, 128, 256, 512, 1024, 2048}
        assert cfg.precision_bits in {4, 6, 8, 12, 16}
        assert 0.0 <= cfg.accuracy_score <= 1.0
        assert cfg.luts_used > 0
        assert cfg.power_used >= 0.0


def test_empty_network_returns_empty_feasible_report() -> None:
    opt = SurrogateSCOptimizer(_target())
    report = opt.optimise([])

    assert report is not None
    assert report.feasible
    assert report.config == {}
    assert report.total_luts == 0
    assert report.total_power_mw == 0.0
    assert report.mean_accuracy == 0.0
    assert report.training_points == 0


def test_critical_path_keeps_accuracy_priority() -> None:
    opt = SurrogateSCOptimizer(_target(max_luts=160_000, max_power_mw=500.0))
    report = opt.optimise(_network())

    assert report is not None
    assert report.feasible
    assert report.config["encoder"].accuracy_score >= report.config["decoder"].accuracy_score


def test_tight_budget_reports_rejected_layers_without_overcommitting() -> None:
    opt = SurrogateSCOptimizer(_target(max_luts=8, max_power_mw=0.001))
    report = opt.optimise(_network())

    assert report is not None
    assert not report.feasible
    assert report.rejected_layers
    assert report.total_luts <= opt.target.budget.max_luts
    assert report.total_power_mw <= opt.target.budget.max_power_mw


def test_benchmark_observation_can_influence_precision_choice() -> None:
    observation = BenchmarkObservation(
        mac_count=256,
        bitstream_length=128,
        decorrelator="LFSR",
        mode="SC",
        precision_bits=8,
        lfsr_polynomial="x16+x15+x13+x4+1",
        luts_used=300,
        power_mw=1.2,
        latency_cycles=128,
        accuracy_score=0.999,
        is_critical_path=True,
    )
    opt = SurrogateSCOptimizer(
        _target(max_luts=20_000, max_power_mw=200.0),
        bitstream_options=(64, 128),
        precision_options=(4, 8),
        observations=(observation,),
    )
    report = opt.optimise([LayerProfile(id="encoder", mac_count=256, is_critical_path=True)])

    assert report is not None
    assert report.feasible
    cfg = report.config["encoder"]
    assert cfg.mode == "SC"
    assert cfg.decorrelator == "LFSR"
    assert cfg.precision_bits == 8
    assert cfg.bitstream_length == 128
    assert cfg.lfsr_polynomial == "x16+x15+x13+x4+1"


def test_surrogate_optimizer_is_deterministic() -> None:
    first = SurrogateSCOptimizer(_target()).optimise(_network())
    second = SurrogateSCOptimizer(_target()).optimise(_network())

    assert first is not None
    assert second is not None
    assert first.config == second.config
    assert first.mean_accuracy == second.mean_accuracy


def test_ridge_surrogate_rejects_non_two_dimensional_inputs() -> None:
    surrogate = _RidgeSurrogate()
    with pytest.raises(ValueError, match="two-dimensional"):
        surrogate.fit(np.ones(3), np.ones((3, 1)))


def test_ridge_surrogate_rejects_row_count_mismatch() -> None:
    surrogate = _RidgeSurrogate()
    with pytest.raises(ValueError, match="row count mismatch"):
        surrogate.fit(np.ones((3, 2)), np.ones((4, 1)))


def test_ridge_surrogate_predict_before_fit_raises() -> None:
    surrogate = _RidgeSurrogate()
    with pytest.raises(RuntimeError, match="not fitted"):
        surrogate.predict(np.ones((1, 2)))


def test_rebalance_greedy_pass_upgrades_low_utility_selection() -> None:
    # Drive the greedy second pass: seed every layer with a sentinel-low utility
    # so _rank_layer_candidates always surfaces a better affordable candidate,
    # exercising the gain-improvement and apply branches of _rebalance.
    opt = SurrogateSCOptimizer(_target())
    network = _network()
    report = opt.optimise(network)
    assert report is not None

    poisoned = {
        layer_id: dataclasses.replace(cfg, utility_score=-1.0e9)
        for layer_id, cfg in report.config.items()
    }
    rebalanced = opt._rebalance(poisoned, network)
    assert any(cfg.utility_score > -1.0e9 for cfg in rebalanced.config.values())
