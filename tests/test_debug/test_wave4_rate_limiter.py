# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRateLimiter from former test_wave4.py

"""Focused suite: TestRateLimiter from former test_wave4.py."""

from __future__ import annotations

from sc_neurocore.debug.hil_client import (
    RateLimiter,
)


class TestRateLimiter:
    """The token-bucket rate limiter."""

    def test_allow(self) -> None:
        """A full bucket allows its burst."""
        rl = RateLimiter(3)
        assert rl.allow()
        assert rl.allow()
        assert rl.allow()
        assert not rl.allow()

    def test_refill(self) -> None:
        """An empty bucket refuses until it refills."""
        rl = RateLimiter(2)
        rl.allow()
        rl.allow()
        rl.refill(1)
        assert rl.allow()
        assert not rl.allow()
