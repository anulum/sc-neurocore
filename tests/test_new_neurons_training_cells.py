# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTrainingCells from former test_new_neurons.py

"""Focused suite: TestTrainingCells from former test_new_neurons.py."""

from __future__ import annotations

from tests.new_neurons_support import *  # noqa: F403

class TestTrainingCells:
    def test_expif_cell(self):
        from sc_neurocore.training.snn_modules import ExpIFCell

        cell = ExpIFCell()
        v = torch.zeros(4)
        spike, v = cell(torch.ones(4) * 2.0, v)
        assert spike.shape == (4,)

    def test_adex_cell(self):
        from sc_neurocore.training.snn_modules import AdExCell

        cell = AdExCell()
        v = torch.zeros(4)
        w = torch.zeros(4)
        spike, v, w = cell(torch.ones(4) * 2.0, v, w)
        assert spike.shape == (4,)

    def test_lapicque_cell(self):
        from sc_neurocore.training.snn_modules import LapicqueCell

        cell = LapicqueCell()
        v = torch.zeros(4)
        spike, v = cell(torch.ones(4) * 5.0, v)
        assert spike.shape == (4,)

    def test_alpha_cell(self):
        from sc_neurocore.training.snn_modules import AlphaCell

        cell = AlphaCell()
        v = torch.zeros(4)
        i_exc = torch.zeros(4)
        i_inh = torch.zeros(4)
        spike, i_exc, i_inh, v = cell(torch.ones(4), torch.zeros(4), i_exc, i_inh, v)
        assert spike.shape == (4,)

    def test_second_order_lif(self):
        from sc_neurocore.training.snn_modules import SecondOrderLIFCell

        cell = SecondOrderLIFCell()
        v = torch.zeros(4)
        a = torch.zeros(4)
        spike, a, v = cell(torch.ones(4) * 2.0, a, v)
        assert spike.shape == (4,)
