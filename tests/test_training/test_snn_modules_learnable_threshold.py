# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLearnableThreshold from former test_snn_modules.py

"""Focused suite: TestLearnableThreshold from former test_snn_modules.py."""

from __future__ import annotations

from tests.test_training.snn_modules_support import *  # noqa: F403


class TestLearnableThreshold:
    def test_threshold_is_parameter(self):
        lif = LIFCell(threshold=1.0, learn_threshold=True)
        param_names = [n for n, _ in lif.named_parameters()]
        assert "_threshold_log" in param_names

    def test_threshold_gradient_flows(self):
        lif = LIFCell(threshold=1.0, learn_threshold=True)
        current = torch.ones(4) * 1.5
        v = torch.zeros(4)
        spike, _ = lif(current, v)
        spike.sum().backward()
        assert lif._threshold_log.grad is not None

    def test_threshold_round_trips(self):
        lif = LIFCell(threshold=2.5, learn_threshold=True)
        assert lif.threshold.item() == pytest.approx(2.5, abs=1e-5)
