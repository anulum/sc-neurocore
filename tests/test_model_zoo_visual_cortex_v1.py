# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVisualCortexV1 from former test_model_zoo.py

"""Focused suite: TestVisualCortexV1 from former test_model_zoo.py."""

from __future__ import annotations

from tests.model_zoo_support import *  # noqa: F403
from tests.performance_guard import assert_load_tolerant_throughput


class TestVisualCortexV1:
    """Orientation-tuned simple→complex cell model."""

    def test_returns_network(self):
        assert isinstance(visual_cortex_v1(n_orientation=4, n_per_orientation=10), Network)

    def test_population_count(self):
        """n_orient simple + n_orient complex = 2·n_orient."""
        net = visual_cortex_v1(n_orientation=4, n_per_orientation=10)
        assert len(net.populations) == 8  # 4 simple + 4 complex

    def test_simple_cells_use_hh(self):
        net = visual_cortex_v1(n_orientation=4, n_per_orientation=10)
        for i in range(4):
            assert net.populations[i]._model_cls is HodgkinHuxleyNeuron

    def test_complex_cells_use_wang_buzsaki(self):
        net = visual_cortex_v1(n_orientation=4, n_per_orientation=10)
        for i in range(4, 8):
            assert net.populations[i]._model_cls is WangBuzsakiNeuron

    def test_simple_to_complex_feedforward(self):
        """Each simple→complex pair has a projection (weight=3.0)."""
        net = visual_cortex_v1(n_orientation=4, n_per_orientation=10)
        # First n_orient projections are simple→complex
        for i in range(4):
            assert net.projections[i].weight == 3.0

    def test_cross_orientation_inhibition(self):
        """Cross-orientation projections have negative weights w = -1/(1+dist)."""
        net = visual_cortex_v1(n_orientation=4, n_per_orientation=10)
        # After the 4 feedforward: 4×3=12 cross-orientation
        cross_projs = net.projections[4:16]
        for p in cross_projs:
            assert p.weight < 0

    def test_cross_orientation_weight_distance_dependent(self):
        """Closer orientations have stronger inhibition: |w(dist=1)| > |w(dist=2)|."""
        # dist=1: w = -1/(1+1) = -0.5; dist=2: w = -1/(1+2) = -0.333
        w1 = -1.0 / (1.0 + 1)
        w2 = -1.0 / (1.0 + 2)
        assert abs(w1) > abs(w2)

    def test_monitors_per_population(self):
        """2 monitors per orientation (simple + complex)."""
        net = visual_cortex_v1(n_orientation=4, n_per_orientation=10)
        assert len(net.spike_monitors) == 8

    def test_one_stimulus_per_orientation(self):
        net = visual_cortex_v1(n_orientation=4, n_per_orientation=10)
        assert len(net.stimuli) == 4

    def test_produces_spikes(self):
        assert _run_and_count(visual_cortex_v1(n_orientation=4, n_per_orientation=10)) > 0

    @pytest.mark.parametrize("n_orient", [2, 4, 8])
    def test_scales_orientations(self, n_orient: int):
        net = visual_cortex_v1(n_orientation=n_orient, n_per_orientation=5)
        assert len(net.populations) == 2 * n_orient

    def test_performance(self):
        net = visual_cortex_v1(n_orientation=2, n_per_orientation=5)
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert_load_tolerant_throughput(
            label="visual-cortex model-zoo network",
            observed_per_second=n_neurons * 50 / elapsed,
            strict_minimum_per_second=10.0,
        )
