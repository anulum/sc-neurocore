# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestALIFCell from former test_snn_modules.py

"""Focused suite: TestALIFCell from former test_snn_modules.py."""

from __future__ import annotations

from tests.test_training.snn_modules_support import *  # noqa: F403


class TestALIFCell:
    def test_adaptation_increases_threshold(self):
        alif = ALIFCell(beta=0.0, threshold=1.0, rho=0.9, beta_adapt=1.0)
        v = torch.zeros(4)
        a = torch.zeros(4)
        current = torch.ones(4) * 2.0
        spike, v_next, a_next = alif(current, v, a)
        # After first spike, a should increase
        assert (a_next > 0).all()

    def test_adaptation_decays_without_spikes(self):
        alif = ALIFCell(beta=0.0, threshold=100.0, rho=0.5)
        v = torch.zeros(4)
        a = torch.ones(4)
        current = torch.zeros(4)
        _, _, a_next = alif(current, v, a)
        # rho=0.5, no spikes: a decays to 0.5
        assert a_next.mean().item() == pytest.approx(0.5, abs=0.01)

    def test_gradient_flows(self):
        alif = ALIFCell()
        current = torch.randn(8, requires_grad=True)
        v = torch.zeros(8)
        a = torch.zeros(8)
        spike, _, _ = alif(current, v, a)
        spike.sum().backward()
        assert current.grad is not None

    def test_output_shapes(self):
        alif = ALIFCell()
        current = torch.randn(2, 8)
        v = torch.zeros(2, 8)
        a = torch.zeros(2, 8)
        spike, v_next, a_next = alif(current, v, a)
        assert spike.shape == (2, 8)
        assert v_next.shape == (2, 8)
        assert a_next.shape == (2, 8)
