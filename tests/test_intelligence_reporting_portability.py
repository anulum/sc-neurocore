# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPortability from former test_intelligence_reporting.py

"""Focused suite: TestPortability from former test_intelligence_reporting.py."""

from __future__ import annotations

from tests.intelligence_reporting_support import *  # noqa: F403

class TestPortability:
    def test_simple_model(self):
        from sc_neurocore.compiler.intelligence import score_portability

        s = score_portability({"v": "a + b"})
        assert s.score > 50
        assert s.compatible_profiles > 0

    def test_complex_model(self):
        from sc_neurocore.compiler.intelligence import score_portability

        s = score_portability({"v": "a*b*c/d*e/f*g*h"})
        assert len(s.blockers) > 0
