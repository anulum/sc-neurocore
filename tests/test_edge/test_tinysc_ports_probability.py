# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestProbability from former test_tinysc_ports.py

"""Focused suite: TestProbability from former test_tinysc_ports.py."""

from __future__ import annotations

from tinysc_ports_support import *  # noqa: F403

class TestProbability:
    def test_all_ones(self):
        assert abs(probability([MASK32], 32) - 1.0) < 1e-6

    def test_all_zeros(self):
        assert abs(probability([0], 32) - 0.0) < 1e-6

    def test_half(self):
        assert abs(probability([0xAAAA_AAAA], 32) - 0.5) < 1e-6

    def test_zero_length(self):
        assert probability([MASK32], 0) == 0.0
