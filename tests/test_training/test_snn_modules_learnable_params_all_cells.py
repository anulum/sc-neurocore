# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLearnableParamsAllCells from former test_snn_modules.py

"""Focused suite: TestLearnableParamsAllCells from former test_snn_modules.py."""

from __future__ import annotations

from tests.test_training.snn_modules_support import *  # noqa: F403

class TestLearnableParamsAllCells:
    """Verify learn_beta/learn_threshold across all cell types (Tier 5.9)."""

    def test_if_learnable_threshold(self):
        cell = IFCell(threshold=1.5, learn_threshold=True)
        assert cell.threshold.item() == pytest.approx(1.5, abs=1e-4)
        x = torch.ones(4) * 2.0
        v = torch.zeros(4)
        spike, _ = cell(x, v)
        spike.sum().backward()
        assert cell._threshold_log.grad is not None

    def test_synaptic_learnable_threshold(self):
        cell = SynapticCell(learn_beta=True, learn_threshold=True)
        assert "_beta_logit" in [n for n, _ in cell.named_parameters()]
        assert "_threshold_log" in [n for n, _ in cell.named_parameters()]
        x = torch.ones(4) * 2.0
        i_syn = torch.zeros(4)
        v = torch.zeros(4)
        spike, _, _ = cell(x, i_syn, v)
        spike.sum().backward()
        assert cell._threshold_log.grad is not None

    def test_expif_learnable(self):
        cell = ExpIFCell(beta=0.8, threshold=1.0, learn_beta=True, learn_threshold=True)
        assert cell.beta.item() == pytest.approx(0.8, abs=1e-4)
        x = torch.ones(4) * 3.0
        v = torch.zeros(4)
        spike, _ = cell(x, v)
        spike.sum().backward()
        assert cell._beta_logit.grad is not None
        assert cell._threshold_log.grad is not None

    def test_adex_learnable(self):
        cell = AdExCell(beta=0.85, threshold=1.0, learn_beta=True, learn_threshold=True)
        assert cell.beta.item() == pytest.approx(0.85, abs=1e-4)
        x = torch.ones(4) * 3.0
        v = torch.zeros(4)
        w = torch.zeros(4)
        spike, _, _ = cell(x, v, w)
        spike.sum().backward()
        assert cell._beta_logit.grad is not None

    def test_lapicque_learnable_threshold(self):
        cell = LapicqueCell(threshold=2.0, learn_threshold=True)
        assert cell.threshold.item() == pytest.approx(2.0, abs=1e-4)
        x = torch.ones(4) * 5.0
        v = torch.zeros(4)
        spike, _ = cell(x, v)
        spike.sum().backward()
        assert cell._threshold_log.grad is not None

    def test_alpha_learnable(self):
        cell = AlphaCell(beta=0.9, threshold=1.0, learn_beta=True, learn_threshold=True)
        assert cell.beta.item() == pytest.approx(0.9, abs=1e-4)
        exc = torch.ones(4) * 2.0
        inh = torch.zeros(4)
        i_exc = torch.zeros(4)
        i_inh = torch.zeros(4)
        v = torch.zeros(4)
        spike, _, _, _ = cell(exc, inh, i_exc, i_inh, v)
        spike.sum().backward()
        assert cell._beta_logit.grad is not None

    def test_second_order_learnable(self):
        cell = SecondOrderLIFCell(beta=0.9, threshold=1.0, learn_beta=True, learn_threshold=True)
        assert cell.beta.item() == pytest.approx(0.9, abs=1e-4)
        x = torch.ones(4) * 3.0
        a = torch.zeros(4)
        v = torch.zeros(4)
        spike, _, _ = cell(x, a, v)
        spike.sum().backward()
        assert cell._beta_logit.grad is not None
        assert cell._threshold_log.grad is not None
