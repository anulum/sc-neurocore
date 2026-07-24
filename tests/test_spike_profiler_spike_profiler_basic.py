# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeProfilerBasic from former test_spike_profiler.py

"""Focused suite: TestSpikeProfilerBasic from former test_spike_profiler.py."""

from __future__ import annotations

from tests.spike_profiler_support import *  # noqa: F403


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
