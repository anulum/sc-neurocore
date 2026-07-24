# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPortabilityManyVariables from former test_intelligence_reporting.py

"""Focused suite: TestPortabilityManyVariables from former test_intelligence_reporting.py."""

from __future__ import annotations

from tests.intelligence_reporting_support import *  # noqa: F403


class TestPortabilityManyVariables:
    """A model with more than four state variables raises the register-file
    blocker that the single-variable portability cases never trigger."""

    def test_many_state_variables_flagged_as_blocker(self):
        from sc_neurocore.compiler.intelligence import score_portability

        eqs = {f"v{i}": "a + b" for i in range(5)}
        s = score_portability(eqs)
        assert any("state variables" in blocker for blocker in s.blockers)
