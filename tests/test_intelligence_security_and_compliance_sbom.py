# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSBOM from former test_intelligence_security_and_compliance.py

"""Focused suite: TestSBOM from former test_intelligence_security_and_compliance.py."""

from __future__ import annotations

from tests.intelligence_security_and_compliance_support import *  # noqa: F403

class TestSBOM:
    def test_basic(self):
        from sc_neurocore.compiler.intelligence import generate_sbom

        s = generate_sbom("sc_lif", "artix7")
        assert s.total_components >= 3
        assert s.format == "CycloneDX"

    def test_with_deps(self):
        from sc_neurocore.compiler.intelligence import generate_sbom

        s = generate_sbom("sc_lif", "artix7", dependencies={"numpy": "1.26.0"})
        assert s.total_components >= 4
