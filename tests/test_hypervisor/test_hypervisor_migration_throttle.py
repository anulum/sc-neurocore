# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMigrationThrottle from former test_hypervisor.py

"""Focused suite: TestMigrationThrottle from former test_hypervisor.py."""

from __future__ import annotations

from sc_neurocore.hypervisor.hypervisor import (
    MigrationThrottle,
)


class TestMigrationThrottle:
    """Rate limiting of tenant migrations."""

    def test_initial_allow(self) -> None:
        """The first migration is allowed."""
        mt = MigrationThrottle(max_per_window=3)
        assert mt.allow() is True

    def test_throttled(self) -> None:
        """Migrations beyond the limit are throttled."""
        mt = MigrationThrottle(max_per_window=2, window_ns=10_000_000_000)
        mt.record()
        mt.record()
        assert mt.allow() is False

    def test_recent_count(self) -> None:
        """The throttle counts its recent migrations."""
        mt = MigrationThrottle()
        mt.record()
        mt.record()
        assert mt.recent_count == 2
