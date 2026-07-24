# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAlphaCell from former test_torch_training.py

"""Focused suite: TestAlphaCell from former test_torch_training.py."""

from __future__ import annotations

from tests.torch_training_support import *  # noqa: F403


class TestAlphaCell:
    def test_excitatory_inhibitory(self):
        cell = AlphaCell()
        exc = torch.tensor([[1.0]])
        inh = torch.tensor([[0.5]])
        i_exc = torch.zeros(1, 1)
        i_inh = torch.zeros(1, 1)
        v = torch.zeros(1, 1)
        spike, ie, ii, v_next = cell(exc, inh, i_exc, i_inh, v)
        assert ie.item() > ii.item()  # more excitation
