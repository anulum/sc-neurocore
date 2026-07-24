# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPopcount from former test_tinysc_ports.py

"""Focused suite: TestPopcount from former test_tinysc_ports.py."""

from __future__ import annotations

from tinysc_ports_support import *  # noqa: F403


class TestPopcount:
    def test_zero(self):
        assert popcount32(0) == 0

    def test_all_ones(self):
        assert popcount32(MASK32) == 32

    def test_alternating(self):
        assert popcount32(0xAAAA_AAAA) == 16

    def test_single_bit(self):
        for i in range(32):
            assert popcount32(1 << i) == 1

    def test_slice(self):
        assert popcount_slice([MASK32, MASK32]) == 64

    def test_slice_empty(self):
        assert popcount_slice([]) == 0
