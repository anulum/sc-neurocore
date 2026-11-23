# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitstreamFirewall from former test_isolation.py

"""Focused suite: TestBitstreamFirewall from former test_isolation.py."""

from __future__ import annotations

from sc_neurocore.hypervisor.isolation import (
    BitstreamFirewall,
    FirewallRule,
)


class TestBitstreamFirewall:
    """Address rules and violations of the bitstream firewall."""

    def test_allow_read(self) -> None:
        """A tenant reads inside its own range."""
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100))
        assert fw.check_access("t0", 0x1050) is True

    def test_deny_out_of_range(self) -> None:
        """An access outside the tenant's range is denied."""
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100))
        assert fw.check_access("t0", 0x2000) is False

    def test_deny_cross_tenant(self) -> None:
        """Another tenant is denied the range."""
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100))
        assert fw.check_access("t1", 0x1050) is False

    def test_deny_write(self) -> None:
        """A write to a read-only range is denied."""
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100, write_allowed=False))
        assert fw.check_access("t0", 0x1050, is_write=True) is False

    def test_deny_read(self) -> None:
        """A read from a range without read access is denied."""
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100, read_allowed=False))
        assert fw.check_access("t0", 0x1050) is False

    def test_violation_logged(self) -> None:
        """A denied access is logged as a violation."""
        fw = BitstreamFirewall()
        fw.check_access("t0", 0x1000)
        assert fw.violation_count == 1

    def test_remove_rules(self) -> None:
        """Removing a tenant's rules denies its accesses."""
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100))
        fw.add_rule(FirewallRule("t1", 0x2000, 0x100))
        removed = fw.remove_tenant_rules("t0")
        assert removed == 1
        assert fw.check_access("t0", 0x1050) is False

    def test_clear_violations(self) -> None:
        """Clearing violations resets their count."""
        fw = BitstreamFirewall()
        fw.check_access("t0", 0x1000)
        fw.clear_violations()
        assert fw.violation_count == 0
