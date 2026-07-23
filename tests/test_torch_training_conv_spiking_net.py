# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConvSpikingNet from former test_torch_training.py

"""Focused suite: TestConvSpikingNet from former test_torch_training.py."""

from __future__ import annotations

from tests.torch_training_support import *  # noqa: F403

class TestConvSpikingNet:
    def test_forward(self):
        net = ConvSpikingNet(n_output=10)
        x = torch.randn(10, 4, 1, 28, 28)  # (T=10, batch=4, C=1, H=28, W=28)
        spike_counts, mem_acc = net(x)
        assert spike_counts.shape == (4, 10)

    def test_to_sc_weights(self):
        net = ConvSpikingNet(n_output=10)
        sc = net.to_sc_weights()
        assert len(sc) == 4  # conv1, conv2, fc1, fc2
