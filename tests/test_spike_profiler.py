# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.profiling.spike_profiler

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.profiling.spike_profiler import (
    SpikeProfiler,
    LayerStats,
    ProfileReport,
    Pathology,
    Severity,
)


def _random_spikes(n_neurons, rate=0.1, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    return (rng.random(n_neurons) < rate).astype(np.int8)


class TestSpikeProfilerBasic:
    def test_empty_report(self):
        p = SpikeProfiler()
        r = p.report()
        assert r.total_steps == 0
        assert r.total_spikes == 0
        assert len(r.pathologies) == 0

    def test_record_single_step(self):
        p = SpikeProfiler()
        spikes = np.array([1, 0, 1, 0, 0], dtype=np.int8)
        p.record_step("layer1", spikes)
        r = p.report()
        assert r.total_steps == 1
        assert r.total_spikes == 2
        assert "layer1" in r.layer_stats
        assert r.layer_stats["layer1"].n_neurons == 5

    def test_record_multiple_steps(self):
        p = SpikeProfiler()
        rng = np.random.RandomState(42)
        for _ in range(100):
            p.record_step("exc", _random_spikes(20, rate=0.1, rng=rng))
        r = p.report()
        stats = r.layer_stats["exc"]
        assert stats.n_steps == 100
        assert stats.n_neurons == 20
        assert stats.firing_rates is not None
        assert 0.0 < stats.firing_rates.mean() < 0.5

    def test_record_with_voltages(self):
        p = SpikeProfiler()
        spikes = np.array([1, 0, 0], dtype=np.int8)
        voltages = np.array([0.8, 0.2, -0.1])
        p.record_step("h", spikes, voltages=voltages)
        r = p.report()
        stats = r.layer_stats["h"]
        assert stats.voltage_mean != 0.0
        assert stats.voltage_std >= 0.0
        assert stats.voltage_min == pytest.approx(-0.1)
        assert stats.voltage_max == pytest.approx(0.8)

    def test_record_with_gradients(self):
        p = SpikeProfiler()
        spikes = np.array([1, 0], dtype=np.int8)
        grads = np.array([0.5, 0.01])
        p.record_step("h", spikes, gradients=grads)
        r = p.report()
        stats = r.layer_stats["h"]
        assert stats.gradient_norm_mean > 0
        assert stats.gradient_norm_max > 0

    def test_batch_input(self):
        p = SpikeProfiler()
        # batch=4, n_neurons=8
        spikes = np.zeros((4, 8), dtype=np.int8)
        spikes[0, 0] = 1
        spikes[2, 3] = 1
        p.record_step("layer", spikes)
        r = p.report()
        assert r.layer_stats["layer"].total_spikes == 2

    def test_multiple_layers(self):
        p = SpikeProfiler()
        rng = np.random.RandomState(0)
        for _ in range(50):
            p.record_step("input", _random_spikes(10, 0.2, rng))
            p.record_step("hidden", _random_spikes(20, 0.1, rng))
            p.record_step("output", _random_spikes(5, 0.05, rng))
        r = p.report()
        assert len(r.layer_stats) == 3
        assert r.total_neurons == 35

    def test_reset(self):
        p = SpikeProfiler()
        p.record_step("h", np.array([1, 0, 1], dtype=np.int8))
        p.reset()
        r = p.report()
        assert r.total_steps == 0


class TestPathologyDetection:
    def test_dead_neurons_critical(self):
        p = SpikeProfiler(dead_threshold=0.01)
        # 80% dead neurons (8 out of 10 never fire)
        for _ in range(100):
            spikes = np.zeros(10, dtype=np.int8)
            spikes[0] = 1
            spikes[1] = 1
            p.record_step("h", spikes)
        r = p.report()
        dead_path = [x for x in r.pathologies if x.category == "dead_neurons"]
        assert len(dead_path) >= 1
        assert dead_path[0].severity == Severity.CRITICAL

    def test_dead_neurons_warning(self):
        p = SpikeProfiler()
        rng = np.random.RandomState(42)
        for _ in range(100):
            spikes = np.zeros(10, dtype=np.int8)
            # 8 neurons fire normally, 2 are dead = 20%
            spikes[:8] = _random_spikes(8, 0.3, rng)
            p.record_step("h", spikes)
        r = p.report()
        dead_path = [x for x in r.pathologies if x.category == "dead_neurons"]
        assert len(dead_path) >= 1
        assert dead_path[0].severity == Severity.WARNING

    def test_saturated_neurons(self):
        p = SpikeProfiler(saturated_threshold=0.95)
        for _ in range(100):
            spikes = np.ones(10, dtype=np.int8)  # All neurons fire every step
            p.record_step("h", spikes)
        r = p.report()
        sat_path = [x for x in r.pathologies if x.category == "saturated_neurons"]
        assert len(sat_path) >= 1

    def test_silent_network(self):
        p = SpikeProfiler()
        for _ in range(20):
            p.record_step("h", np.zeros(10, dtype=np.int8))
        r = p.report()
        silent = [x for x in r.pathologies if x.category == "silent_network"]
        assert len(silent) >= 1
        assert silent[0].severity == Severity.CRITICAL

    def test_voltage_collapse(self):
        p = SpikeProfiler()
        for _ in range(20):
            p.record_step("h", np.zeros(5, dtype=np.int8), voltages=np.zeros(5))
        r = p.report()
        collapse = [x for x in r.pathologies if x.category == "voltage_collapse"]
        assert len(collapse) >= 1

    def test_gradient_explosion(self):
        p = SpikeProfiler(gradient_explosion_ratio=10.0)
        for _ in range(10):
            p.record_step("h", np.ones(5, dtype=np.int8), gradients=np.ones(5) * 0.01)
        # One step with huge gradient
        p.record_step("h", np.ones(5, dtype=np.int8), gradients=np.ones(5) * 1000.0)
        r = p.report()
        explode = [x for x in r.pathologies if x.category == "gradient_explosion"]
        assert len(explode) >= 1
        assert explode[0].severity == Severity.CRITICAL

    def test_gradient_vanishing(self):
        p = SpikeProfiler()
        for _ in range(10):
            p.record_step("layer1", np.ones(5, dtype=np.int8), gradients=np.ones(5) * 10.0)
            p.record_step("layer2", np.ones(5, dtype=np.int8), gradients=np.ones(5) * 0.001)
        r = p.report()
        vanish = [x for x in r.pathologies if x.category == "gradient_vanishing"]
        assert len(vanish) >= 1

    def test_healthy_network_no_pathologies(self):
        p = SpikeProfiler()
        rng = np.random.RandomState(42)
        for _ in range(100):
            p.record_step(
                "h",
                _random_spikes(20, 0.15, rng),
                voltages=rng.randn(20) * 0.3,
            )
        r = p.report()
        critical = [x for x in r.pathologies if x.severity == Severity.CRITICAL]
        assert len(critical) == 0


class TestProfileReport:
    def test_summary_format(self):
        p = SpikeProfiler()
        rng = np.random.RandomState(0)
        for _ in range(10):
            p.record_step("h", _random_spikes(8, 0.2, rng))
        r = p.report()
        s = r.summary()
        assert "SpikeProfiler Report" in s
        assert "h:" in s

    def test_has_critical(self):
        r = ProfileReport(
            pathologies=[
                Pathology(Severity.CRITICAL, "test", "l", "msg", "fix"),
            ]
        )
        assert r.has_critical is True

    def test_no_critical(self):
        r = ProfileReport(
            pathologies=[
                Pathology(Severity.WARNING, "test", "l", "msg", "fix"),
            ]
        )
        assert r.has_critical is False

    def test_empty_has_critical(self):
        r = ProfileReport()
        assert r.has_critical is False


class TestLayerStats:
    def test_fields(self):
        s = LayerStats(name="test", n_neurons=10, n_steps=5)
        assert s.dead_neuron_count == 0
        assert s.estimated_syn_ops == 0


class TestPathology:
    def test_fields(self):
        p = Pathology(
            severity=Severity.WARNING,
            category="dead_neurons",
            layer="hidden",
            message="50% dead",
            suggestion="lower threshold",
            metric_value=0.5,
        )
        assert p.severity == Severity.WARNING
        assert p.metric_value == 0.5
