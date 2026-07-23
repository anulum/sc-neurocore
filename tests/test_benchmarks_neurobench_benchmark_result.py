# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBenchmarkResult from former test_benchmarks_neurobench.py

"""Focused suite: TestBenchmarkResult from former test_benchmarks_neurobench.py."""

from __future__ import annotations

from tests.benchmarks_neurobench_support import *  # noqa: F403

class TestBenchmarkResult:
    def test_to_json(self):
        r = BenchmarkResult(
            task="mnist",
            model="test_snn",
            accuracy=0.95,
            total_parameters=1000,
            synaptic_operations=50000,
            activation_sparsity=0.8,
            total_spikes=200,
            timesteps=16,
            latency_ms=5.0,
        )
        j = json.loads(r.to_neurobench_json())
        assert j["task"] == "mnist"
        assert j["metrics"]["correctness"]["accuracy"] == 0.95
        assert j["metrics"]["complexity"]["total_parameters"] == 1000
        assert j["framework"] == "sc-neurocore"

    def test_to_json_with_energy(self):
        r = BenchmarkResult(
            task="shd",
            model="m",
            accuracy=0.8,
            total_parameters=500,
            synaptic_operations=1000,
            activation_sparsity=0.5,
            total_spikes=100,
            timesteps=8,
            latency_ms=1.0,
            energy_nj=42.0,
        )
        j = json.loads(r.to_neurobench_json())
        assert j["metrics"]["system"]["energy_nj"] == 42.0

    def test_summary(self):
        r = BenchmarkResult(
            task="dvs_gesture",
            model="snn_v1",
            accuracy=0.92,
            total_parameters=5000,
            synaptic_operations=100000,
            activation_sparsity=0.9,
            total_spikes=500,
            timesteps=32,
            latency_ms=10.0,
        )
        s = r.summary()
        assert "dvs_gesture" in s
        assert "0.9200" in s
        assert "snn_v1" in s

    def test_summary_with_energy(self):
        r = BenchmarkResult(
            task="t",
            model="m",
            accuracy=0.5,
            total_parameters=1,
            synaptic_operations=1,
            activation_sparsity=0.1,
            total_spikes=1,
            timesteps=1,
            latency_ms=1.0,
            energy_nj=99.5,
        )
        s = r.summary()
        assert "99.50" in s

    def test_extra_field(self):
        r = BenchmarkResult(
            task="t",
            model="m",
            accuracy=0.5,
            total_parameters=1,
            synaptic_operations=1,
            activation_sparsity=0.1,
            total_spikes=1,
            timesteps=1,
            latency_ms=1.0,
            extra={"custom": {"val": 42}},
        )
        j = json.loads(r.to_neurobench_json())
        assert j["metrics"]["custom"]["val"] == 42
