# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestComputeMetrics from former test_benchmarks_neurobench.py

"""Focused suite: TestComputeMetrics from former test_benchmarks_neurobench.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.benchmarks import (
    compute_metrics,
)


class TestComputeMetrics:
    """Computing NeuroBench metrics."""

    def test_perfect_accuracy(self) -> None:
        """Matching predictions give an accuracy of 1."""
        preds = np.array([0, 1, 2, 3])
        targets = np.array([0, 1, 2, 3])
        r = compute_metrics(preds, targets)
        assert r.accuracy == 1.0

    def test_zero_accuracy(self) -> None:
        """Wrong predictions give an accuracy of 0."""
        preds = np.array([1, 2, 3, 0])
        targets = np.array([0, 1, 2, 3])
        r = compute_metrics(preds, targets)
        assert r.accuracy == 0.0

    def test_with_spike_counts(self) -> None:
        """Spike counts give a total and a sparsity in [0, 1]."""
        preds = np.array([0, 1, 0, 1])
        targets = np.array([0, 1, 0, 1])
        spike_counts = np.array([10, 20, 15, 25])
        weights = [np.random.randn(10, 10)]
        r = compute_metrics(preds, targets, spike_counts=spike_counts, weights=weights, timesteps=8)
        assert r.total_spikes == 70
        assert r.activation_sparsity >= 0
        assert r.activation_sparsity <= 1.0

    def test_no_weights(self) -> None:
        """No weights give no parameters or operations."""
        r = compute_metrics(np.array([0]), np.array([0]))
        assert r.total_parameters == 0
        assert r.synaptic_operations == 0

    def test_custom_task_model(self) -> None:
        """The task and model names are kept."""
        r = compute_metrics(
            np.array([0]),
            np.array([0]),
            task="keyword_spotting",
            model="my_snn",
        )
        assert r.task == "keyword_spotting"
        assert r.model == "my_snn"

    def test_no_spike_counts(self) -> None:
        """No spike counts give zero spikes and sparsity."""
        r = compute_metrics(np.array([0, 1]), np.array([0, 1]))
        assert r.total_spikes == 0
        assert r.activation_sparsity == 0.0
