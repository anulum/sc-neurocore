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
    """signed_q: the signed fixed-point code of a real value."""

    def test_positive_within_range(self) -> None:
        """1.0 encodes to 256 in Q8.8."""
        assert signed_q(Q, 1.0) == 256

    def test_negative_two_complement(self) -> None:
        """-65.0 encodes to the signed code -16640 in Q8.8."""
        # -65.0 in Q8.8 → -16640, reinterpreted signed
        assert signed_q(Q, -65.0) == -16640

    def test_wraps_when_out_of_range(self) -> None:
        """200.0 exceeds Q8.8 and wraps to a negative code."""
        # +200 exceeds Q8.8 max (127.996); the pattern wraps to a negative value
        assert signed_q(Q, 200.0) < 0
