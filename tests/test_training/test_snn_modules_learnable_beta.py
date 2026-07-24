# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLearnableBeta from former test_snn_modules.py

"""Focused suite: TestLearnableBeta from former test_snn_modules.py."""

from __future__ import annotations

from tests.test_training.snn_modules_support import *  # noqa: F403


class TestLearnableBeta:
    def test_beta_is_parameter(self):
        lif = LIFCell(beta=0.9, learn_beta=True)
        param_names = [n for n, _ in lif.named_parameters()]
        assert "_beta_logit" in param_names

    def test_beta_in_valid_range(self):
        lif = LIFCell(beta=0.9, learn_beta=True)
        assert 0 < lif.beta.item() < 1

    def test_beta_gradient_flows(self):
        lif = LIFCell(beta=0.9, learn_beta=True)
        current = torch.ones(4) * 2.0
        v = torch.ones(4) * 0.5
        spike, v_next = lif(current, v)
        (spike.sum() + v_next.sum()).backward()
        assert lif._beta_logit.grad is not None
        assert lif._beta_logit.grad.abs().item() > 0

    def test_beta_round_trips(self):
        lif = LIFCell(beta=0.85, learn_beta=True)
        assert lif.beta.item() == pytest.approx(0.85, abs=1e-5)
