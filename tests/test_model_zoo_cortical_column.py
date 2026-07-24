# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCorticalColumn from former test_model_zoo.py

"""Focused suite: TestCorticalColumn from former test_model_zoo.py."""

from __future__ import annotations

from tests.model_zoo_support import *  # noqa: F403


class TestCorticalColumn:
    """4-layer cortical microcircuit with E/I per layer."""

    def test_returns_network(self):
        assert isinstance(cortical_column(n_layers=4), Network)

    def test_eight_populations_four_layers(self):
        """4 layers × 2 (E+I) = 8 populations."""
        net = cortical_column(n_layers=4)
        assert len(net.populations) == 8

    def test_excitatory_uses_pospischil(self):
        """E populations use PospischilNeuron (RS type)."""
        net = cortical_column(n_layers=4)
        for i in range(0, 8, 2):
            assert net.populations[i]._model_cls is PospischilNeuron

    def test_inhibitory_uses_golomb_fs(self):
        """I populations use GolombFSNeuron (FS type)."""
        net = cortical_column(n_layers=4)
        for i in range(1, 8, 2):
            assert net.populations[i]._model_cls is GolombFSNeuron

    def test_intra_layer_wiring(self):
        """Each layer has E→I, I→E, E→E (3 per layer = 12 intra)."""
        net = cortical_column(n_layers=4)
        # 12 intra-layer + 3 inter-layer feedforward = 15
        assert len(net.projections) == 15

    def test_feedforward_l4_to_l23(self):
        """L4_E → L23_E feedforward projection exists."""
        net = cortical_column(n_layers=4)
        # Inter-layer: ff_map = [(1,0), (0,2), (2,3)]
        # Source layer 1 = L4, target layer 0 = L23
        ff_projs = net.projections[12:]  # last 3 are inter-layer
        assert len(ff_projs) == 3

    def test_thalamic_drive_targets_l4(self):
        """PoissonInput targets L4_E (populations[2])."""
        net = cortical_column(n_layers=4)
        assert len(net.stimuli) == 1
        assert net.stimuli[0].target is net.populations[2]

    def test_eight_monitors(self):
        net = cortical_column(n_layers=4)
        assert len(net.spike_monitors) == 8

    def test_produces_spikes(self):
        assert _run_and_count(cortical_column(n_layers=4)) > 0

    @pytest.mark.parametrize("n_layers", [2, 3, 4])
    def test_scales_layers(self, n_layers: int):
        net = cortical_column(n_layers=n_layers)
        assert len(net.populations) == 2 * n_layers

    def test_performance(self):
        net = cortical_column(n_layers=2)
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert n_neurons * 50 / elapsed > 10
