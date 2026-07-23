# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestUtilities from former test_torch_training.py

"""Focused suite: TestUtilities from former test_torch_training.py."""

from __future__ import annotations

from tests.torch_training_support import *  # noqa: F403

class TestUtilities:
    def test_model_info(self):
        net = SpikingNet(n_input=16, n_hidden=32, n_output=5)
        info = model_info(net)
        assert info["total_params"] > 0
        assert info["spiking_cells"] > 0
        assert "LIFCell" in info["cell_types"]

    def test_population_decode(self):
        counts = torch.tensor([[0.0, 1.0, 5.0, 0.0]])  # peak at index 2
        decoded = population_decode(counts)
        # weights = [0, 1/6, 5/6, 0], preferred = [0,1,2,3]
        # decoded = 1/6*1 + 5/6*2 = 11/6 ≈ 1.833
        assert decoded.item() == pytest.approx(11 / 6, abs=0.01)

    def test_population_decode_with_preferred(self):
        counts = torch.tensor([[0.0, 0.0, 1.0]])
        preferred = torch.tensor([0.0, 45.0, 90.0])
        decoded = population_decode(counts, preferred)
        assert decoded.item() == pytest.approx(90.0)

    def test_spike_monitor(self):
        net = SpikingNet(n_input=16, n_hidden=32, n_output=5, n_layers=1)
        mon = SpikeMonitor(net)
        x = torch.randn(5, 2, 16)
        net(x)
        assert len(mon.layer_names) > 0
        for name in mon.layer_names:
            data = mon.get(name)
            assert data is not None
        mon.reset()
        for name in mon.layer_names:
            assert mon.get(name) is None
        mon.remove()

    def test_reset_states(self):
        net = SpikingNet(n_input=16, n_hidden=32, n_output=5)
        mon = SpikeMonitor(net)
        reset_states([mon])
        reset_states(None)  # should not raise
