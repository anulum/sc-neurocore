# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRateLimiter from former test_wave4.py

"""Focused suite: TestRateLimiter from former test_wave4.py."""

from __future__ import annotations

from wave4_support import *  # noqa: F403


class TestRateLimiter:
    def test_allow(self):
        rl = RateLimiter(3)
        assert rl.allow()
        assert rl.allow()
        assert rl.allow()
        assert not rl.allow()

    def test_refill(self):
        rl = RateLimiter(2)
        rl.allow()
        rl.allow()
        rl.refill(1)
        assert rl.allow()
        assert not rl.allow()
