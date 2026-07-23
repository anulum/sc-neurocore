# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSignedQ from former test_c_fixed_emitter.py

"""Focused suite: TestSignedQ from former test_c_fixed_emitter.py."""

from __future__ import annotations

from tests.c_fixed_emitter_support import *  # noqa: F403

class TestSignedQ:
    def test_positive_within_range(self):
        assert signed_q(Q, 1.0) == 256

    def test_negative_two_complement(self):
        # -65.0 in Q8.8 → -16640, reinterpreted signed
        assert signed_q(Q, -65.0) == -16640

    def test_wraps_when_out_of_range(self):
        # +200 exceeds Q8.8 max (127.996); the pattern wraps to a negative value
        assert signed_q(Q, 200.0) < 0
