# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLfsr16 from former test_tinysc_ports.py

"""Focused suite: TestLfsr16 from former test_tinysc_ports.py."""

from __future__ import annotations

from tinysc_ports_support import *  # noqa: F403


class TestLfsr16:
    def test_nonzero_seed(self):
        lfsr = Lfsr16(0xACE1)
        assert lfsr.reg != 0

    def test_step_changes_state(self):
        lfsr = Lfsr16(0xACE1)
        s0 = lfsr.reg
        lfsr.step()
        assert lfsr.reg != s0

    def test_encode_length(self):
        lfsr = Lfsr16(0xACE1)
        words = lfsr.encode(32768, 1024)
        assert len(words) == 32  # 1024 / 32 = 32 words

    def test_encode_float_half(self):
        lfsr = Lfsr16(0xACE1)
        words = lfsr.encode_float(0.5, 1024)
        pc = popcount_slice(words)
        assert 400 < pc < 600  # ~50% ± tolerance

    def test_zero_seed_uses_default(self):
        lfsr = Lfsr16(0)
        assert lfsr.reg == 0xACE1

    def test_period_uniqueness(self):
        lfsr = Lfsr16(0xACE1)
        seen = set()
        for _ in range(1000):
            seen.add(lfsr.step())
        assert len(seen) == 1000  # should all be unique in first 1000
