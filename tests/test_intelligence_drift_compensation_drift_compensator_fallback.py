# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDriftCompensatorFallback from former test_intelligence_drift_compensation.py

"""Focused suite: TestDriftCompensatorFallback from former test_intelligence_drift_compensation.py."""

from __future__ import annotations

from tests.intelligence_drift_compensation_support import *  # noqa: F403


class TestDriftCompensatorFallback:
    """A non-positive drift rate has no tolerance horizon, so the refresh
    interval falls back to the fixed ceiling instead of dividing by zero."""

    def test_non_positive_drift_uses_fallback_refresh(self) -> None:
        from sc_neurocore.compiler.intelligence import generate_drift_compensator

        c = generate_drift_compensator("sc_lif", drift_rate_per_day=0.0)
        assert c.refresh_interval_ms == round(1e9, 2)
