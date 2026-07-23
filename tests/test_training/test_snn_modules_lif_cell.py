# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLIFCell from former test_snn_modules.py

"""Focused suite: TestLIFCell from former test_snn_modules.py."""

from __future__ import annotations

from tests.test_training.snn_modules_support import *  # noqa: F403

class TestLIFCell:
    def test_above_threshold_spikes(self):
        lif = LIFCell(beta=0.0, threshold=0.5)
        v = torch.zeros(4)
        spike, _ = lif(torch.ones(4), v)
        assert spike.sum().item() == 4

    def test_below_threshold_silent(self):
        lif = LIFCell(beta=0.0, threshold=2.0)
        v = torch.zeros(4)
        spike, _ = lif(torch.ones(4), v)
        assert spike.sum().item() == 0

    def test_membrane_reset_after_spike(self):
        lif = LIFCell(beta=0.0, threshold=1.0)
        spike, v = lif(torch.tensor([2.0]), torch.zeros(1))
        assert spike.item() == 1.0
        assert v.item() == pytest.approx(1.0)

    def test_leak_decay(self):
        lif = LIFCell(beta=0.5, threshold=10.0)
        _, v1 = lif(torch.tensor([0.0]), torch.tensor([4.0]))
        assert v1.item() == pytest.approx(2.0)

    def test_gradient_flows(self):
        lif = LIFCell()
        current = torch.randn(8, requires_grad=True)
        v = torch.zeros(8)
        spike, _ = lif(current, v)
        spike.sum().backward()
        assert current.grad is not None

    def test_alternate_surrogate(self):
        lif = LIFCell(surrogate_fn=superspike)
        spike, _ = lif(torch.tensor([2.0]), torch.zeros(1))
        assert spike.item() == 1.0
