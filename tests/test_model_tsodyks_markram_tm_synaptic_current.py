# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTMSynapticCurrent from former test_model_tsodyks_markram.py

"""Focused suite: TestTMSynapticCurrent from former test_model_tsodyks_markram.py."""

from __future__ import annotations

from tests.model_tsodyks_markram_support import *  # noqa: F403

class TestTMSynapticCurrent:
    def test_isyn_on_presyn_spike(self):
        """I_syn = A · u · x when presynaptic spike occurs."""
        n = TsodyksMarkramNeuron()
        # At first spike: u = u_se + U*(1-u) = 0.2 + 0.2*0.8 = 0.36
        # x starts at 1.0
        # i_syn = 50 * 0.36 * 1.0 = 18.0
        # This drives V
        v_before = n.v
        n.step(0.0, presynaptic_spike=True)
        # V should have increased from i_syn
        assert n.v > v_before

    def test_no_isyn_without_presyn(self):
        """Without presynaptic spike, I_syn = 0."""
        n = TsodyksMarkramNeuron()
        n.step(0.0, presynaptic_spike=False)
        # V should have moved only from leak (toward rest)
        assert abs(n.v - n.v_rest) < 0.01
