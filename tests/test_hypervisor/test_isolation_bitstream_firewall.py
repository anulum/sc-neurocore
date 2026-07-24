# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitstreamFirewall from former test_isolation.py

"""Focused suite: TestBitstreamFirewall from former test_isolation.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from isolation_support import *  # noqa: F403


class TestBitstreamFirewall:
    def test_allow_read(self) -> None:
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100))
        assert fw.check_access("t0", 0x1050) is True

    def test_deny_out_of_range(self) -> None:
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100))
        assert fw.check_access("t0", 0x2000) is False

    def test_deny_cross_tenant(self) -> None:
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100))
        assert fw.check_access("t1", 0x1050) is False

    def test_deny_write(self) -> None:
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100, write_allowed=False))
        assert fw.check_access("t0", 0x1050, is_write=True) is False

    def test_deny_read(self) -> None:
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100, read_allowed=False))
        assert fw.check_access("t0", 0x1050) is False

    def test_violation_logged(self) -> None:
        fw = BitstreamFirewall()
        fw.check_access("t0", 0x1000)
        assert fw.violation_count == 1

    def test_remove_rules(self) -> None:
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100))
        fw.add_rule(FirewallRule("t1", 0x2000, 0x100))
        removed = fw.remove_tenant_rules("t0")
        assert removed == 1
        assert fw.check_access("t0", 0x1050) is False

    def test_clear_violations(self) -> None:
        fw = BitstreamFirewall()
        fw.check_access("t0", 0x1000)
        fw.clear_violations()
        assert fw.violation_count == 0
