# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIsolationVerification from former test_isolation.py

"""Focused suite: TestIsolationVerification from former test_isolation.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
from isolation_support import *  # noqa: F403

class TestIsolationVerification:
    def test_no_overlap(self) -> None:
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100))
        fw.add_rule(FirewallRule("t1", 0x2000, 0x100))
        violations = verify_isolation(fw, {})
        assert violations == []

    def test_overlap_detected(self) -> None:
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x200))
        fw.add_rule(FirewallRule("t1", 0x1100, 0x200))
        violations = verify_isolation(fw, {})
        assert len(violations) == 1
        assert "overlap" in violations[0]
