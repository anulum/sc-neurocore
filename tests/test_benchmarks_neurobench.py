# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.benchmarks

from __future__ import annotations

import json

import numpy as np
import pytest

from sc_neurocore.benchmarks import compute_metrics, BenchmarkResult, TASKS
from sc_neurocore.benchmarks.tasks import BenchmarkTask


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


class TestComputeMetrics:
    def test_perfect_accuracy(self):
        preds = np.array([0, 1, 2, 3])
        targets = np.array([0, 1, 2, 3])
        r = compute_metrics(preds, targets)
        assert r.accuracy == 1.0

    def test_zero_accuracy(self):
        preds = np.array([1, 2, 3, 0])
        targets = np.array([0, 1, 2, 3])
        r = compute_metrics(preds, targets)
        assert r.accuracy == 0.0

    def test_with_spike_counts(self):
        preds = np.array([0, 1, 0, 1])
        targets = np.array([0, 1, 0, 1])
        spike_counts = np.array([10, 20, 15, 25])
        weights = [np.random.randn(10, 10)]
        r = compute_metrics(preds, targets, spike_counts=spike_counts, weights=weights, timesteps=8)
        assert r.total_spikes == 70
        assert r.activation_sparsity >= 0
        assert r.activation_sparsity <= 1.0

    def test_no_weights(self):
        r = compute_metrics(np.array([0]), np.array([0]))
        assert r.total_parameters == 0
        assert r.synaptic_operations == 0

    def test_custom_task_model(self):
        r = compute_metrics(
            np.array([0]),
            np.array([0]),
            task="keyword_spotting",
            model="my_snn",
        )
        assert r.task == "keyword_spotting"
        assert r.model == "my_snn"

    def test_no_spike_counts(self):
        r = compute_metrics(np.array([0, 1]), np.array([0, 1]))
        assert r.total_spikes == 0
        assert r.activation_sparsity == 0.0


class TestTasks:
    def test_task_registry(self):
        assert len(TASKS) >= 5
        assert "mnist" in TASKS
        assert "dvs_gesture" in TASKS
        assert "keyword_spotting" in TASKS

    def test_task_fields(self):
        t = TASKS["mnist"]
        assert isinstance(t, BenchmarkTask)
        assert t.n_classes == 10
        assert t.metric == "accuracy"
        assert t.input_shape == (784,)
        assert t.baseline_accuracy > 0

    def test_all_tasks_frozen(self):
        for name, task in TASKS.items():
            with pytest.raises(AttributeError):
                task.name = "mutated"

    def test_shd_task(self):
        t = TASKS["shd"]
        assert t.n_classes == 20
        assert t.neurobench_id == "shd"

    def test_heartbeat_task(self):
        t = TASKS["heartbeat_anomaly"]
        assert t.n_classes == 2
