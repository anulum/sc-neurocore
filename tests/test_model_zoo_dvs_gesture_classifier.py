# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDVSGestureClassifier from former test_model_zoo.py

"""Focused suite: TestDVSGestureClassifier from former test_model_zoo.py."""

from __future__ import annotations

import pytest
import time
from sc_neurocore.model_zoo import (
    dvs_gesture_classifier,
)
from sc_neurocore.network.network import (
    Network,
)
from tests.model_zoo_support import (
    _run_and_count,
    _total_neurons,
)
from tests.performance_guard import assert_load_tolerant_throughput


class TestDVSGestureClassifier:
    """256-256-11 event-camera gesture SNN."""

    def test_returns_network(self) -> None:
        """The model returns network."""
        assert isinstance(dvs_gesture_classifier(n_classes=4), Network)

    def test_topology(self) -> None:
        """The classifier has input, hidden and output populations of the stated sizes."""
        net = dvs_gesture_classifier(n_classes=4)
        assert len(net.populations) == 3
        assert net.populations[0].n == 256  # input
        assert net.populations[1].n == 256  # hidden
        assert net.populations[2].n == 4  # output (parameterised)

    def test_two_projections(self) -> None:
        """The classifier has two projections."""
        net = dvs_gesture_classifier(n_classes=4)
        assert len(net.projections) == 2

    def test_single_monitor(self) -> None:
        """The classifier monitors only its gesture output."""
        net = dvs_gesture_classifier(n_classes=4)
        assert len(net.spike_monitors) == 1
        assert "gesture" in net.spike_monitors[0].label

    def test_produces_spikes(self) -> None:
        """The model produces spikes."""
        assert _run_and_count(dvs_gesture_classifier(n_classes=4)) > 0

    @pytest.mark.parametrize("n_classes", [4, 8, 11])
    def test_scales_output(self, n_classes: int) -> None:
        """The model scales output."""
        net = dvs_gesture_classifier(n_classes=n_classes)
        assert net.populations[2].n == n_classes

    def test_performance(self) -> None:
        """The configuration simulates 50 ms above its load-tolerant throughput floor."""
        net = dvs_gesture_classifier(n_classes=4)
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert_load_tolerant_throughput(
            label="DVS model-zoo network",
            observed_per_second=n_neurons * 50 / elapsed,
            strict_minimum_per_second=100.0,
        )
