# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTanhFSM from former test_utils_extended.py

"""Focused suite: TestTanhFSM from former test_utils_extended.py."""

from __future__ import annotations

from tests.utils_extended_support import *  # noqa: F403

class TestTanhFSM:
    def test_initial_state(self):
        fsm = TanhFSM(states=16)
        assert fsm.state == 8  # N/2

    def test_all_ones_saturates_high(self):
        """All-ones input should drive state to max and output 1."""
        fsm = TanhFSM(states=8)
        bs = np.ones(100, dtype=np.uint8)
        out = fsm.process(bs)
        # After enough 1s, state saturates at 7, output always 1
        assert out[-1] == 1
        # Last 50 outputs should all be 1
        assert np.all(out[-50:] == 1)

    def test_all_zeros_saturates_low(self):
        """All-zeros input should drive state to 0 and output 0."""
        fsm = TanhFSM(states=8)
        bs = np.zeros(100, dtype=np.uint8)
        out = fsm.process(bs)
        assert out[-1] == 0
        assert np.all(out[-50:] == 0)

    def test_balanced_input(self):
        """p=0.5 input should give ~0.5 output probability."""
        rng = np.random.default_rng(42)
        fsm = TanhFSM(states=16)
        bs = (rng.random(4096) < 0.5).astype(np.uint8)
        out = fsm.process(bs)
        # Output probability should be near 0.5
        assert out.mean() == pytest.approx(0.5, abs=0.1)

    def test_high_input_bias(self):
        """High-probability input should produce high-probability output."""
        rng = np.random.default_rng(0)
        fsm = TanhFSM(states=16)
        bs = (rng.random(4096) < 0.9).astype(np.uint8)
        out = fsm.process(bs)
        assert out.mean() > 0.7

    def test_process_returns_correct_length(self):
        fsm = TanhFSM(states=8)
        bs = np.ones(123, dtype=np.uint8)
        out = fsm.process(bs)
        assert len(out) == 123

    def test_output_is_binary(self):
        rng = np.random.default_rng(0)
        fsm = TanhFSM(states=8)
        bs = (rng.random(200) < 0.6).astype(np.uint8)
        out = fsm.process(bs)
        assert set(np.unique(out)).issubset({0, 1})
