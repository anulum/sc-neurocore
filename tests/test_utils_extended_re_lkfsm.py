# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestReLKFSM from former test_utils_extended.py

"""Focused suite: TestReLKFSM from former test_utils_extended.py."""

from __future__ import annotations

from tests.utils_extended_support import *  # noqa: F403

class TestReLKFSM:
    def test_initial_state_zero(self):
        fsm = ReLKFSM(states=16)
        assert fsm.state == 0

    def test_all_zeros_stays_off(self):
        """All-zeros input keeps state at 0, output always 0."""
        fsm = ReLKFSM(states=8)
        bs = np.zeros(100, dtype=np.uint8)
        out = fsm.process(bs)
        assert np.all(out == 0)

    def test_all_ones_turns_on(self):
        """All-ones input drives state above 0, output becomes 1."""
        fsm = ReLKFSM(states=8)
        bs = np.ones(100, dtype=np.uint8)
        out = fsm.process(bs)
        # First step: state goes to 1, output is 1
        assert out[0] == 1
        assert np.all(out == 1)

    def test_relu_like_behavior(self):
        """Low input (p<0.5) should produce lower output than high input (p>0.5)."""
        rng = np.random.default_rng(42)
        fsm_low = ReLKFSM(states=16)
        fsm_high = ReLKFSM(states=16)
        bs_low = (rng.random(4096) < 0.2).astype(np.uint8)
        bs_high = (rng.random(4096) < 0.8).astype(np.uint8)
        out_low = fsm_low.process(bs_low)
        out_high = fsm_high.process(bs_high)
        assert out_high.mean() > out_low.mean()

    def test_step_transitions(self):
        """Verify individual step state transitions."""
        fsm = ReLKFSM(states=4)
        # state=0, input 0 -> state stays 0, output 0
        assert fsm.step(0) == 0
        assert fsm.state == 0
        # state=0, input 1 -> state 1, output 1
        assert fsm.step(1) == 1
        assert fsm.state == 1
        # state=1, input 0 -> state 0, output 0
        assert fsm.step(0) == 0
        assert fsm.state == 0
