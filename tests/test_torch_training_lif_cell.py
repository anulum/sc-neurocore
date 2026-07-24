# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLIFCell from former test_torch_training.py

"""Focused suite: TestLIFCell from former test_torch_training.py."""

from __future__ import annotations

from tests.torch_training_support import *  # noqa: F403


class TestLIFCell:
    def test_forward_shape(self):
        cell = LIFCell(beta=0.9)
        current = torch.randn(8, 64)
        v = torch.zeros(8, 64)
        spike, v_next = cell(current, v)
        assert spike.shape == (8, 64)
        assert v_next.shape == (8, 64)

    def test_spikes_binary(self):
        cell = LIFCell(beta=0.9, threshold=0.5)
        current = torch.randn(32, 16)
        v = torch.zeros(32, 16)
        spike, _ = cell(current, v)
        assert set(spike.unique().tolist()).issubset({0.0, 1.0})

    def test_membrane_reset(self):
        cell = LIFCell(beta=0.9, threshold=1.0)
        current = torch.tensor([[2.0]])  # above threshold
        v = torch.zeros(1, 1)
        spike, v_next = cell(current, v)
        assert spike.item() == 1.0
        assert v_next.item() < 2.0  # reset subtracts threshold

    def test_learnable_beta(self):
        cell = LIFCell(beta=0.9, learn_beta=True)
        assert any("beta_logit" in n for n, _ in cell.named_parameters())

    def test_learnable_threshold(self):
        cell = LIFCell(threshold=1.0, learn_threshold=True)
        assert any("threshold_log" in n for n, _ in cell.named_parameters())

    def test_gradient_flows(self):
        cell = LIFCell(beta=0.9)
        w = torch.randn(16, 32, requires_grad=True)
        x = torch.randn(4, 32)
        current = x @ w.T
        v = torch.zeros(4, 16)
        spike, _ = cell(current, v)
        loss = spike.sum()
        loss.backward()
        assert w.grad is not None
