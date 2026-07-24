# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLicenseChecker from former test_intelligence_security_and_compliance.py

"""Focused suite: TestLicenseChecker from former test_intelligence_security_and_compliance.py."""

from __future__ import annotations

from tests.intelligence_security_and_compliance_support import *  # noqa: F403


class TestLicenseChecker:
    def test_compatible(self):
        from sc_neurocore.compiler.intelligence import check_license_compliance

        r = check_license_compliance("AGPL-3.0", {"numpy": "BSD-3"})
        assert r.compatible is True

    def test_conflict(self):
        from sc_neurocore.compiler.intelligence import check_license_compliance

        r = check_license_compliance("MIT", {"gpl_lib": "GPL-3.0"})
        assert r.compatible is False
        assert len(r.conflicts) == 1
