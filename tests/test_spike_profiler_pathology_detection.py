# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPathologyDetection from former test_spike_profiler.py

"""Focused suite: TestPathologyDetection from former test_spike_profiler.py."""

from __future__ import annotations

from tests.spike_profiler_support import *  # noqa: F403


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
