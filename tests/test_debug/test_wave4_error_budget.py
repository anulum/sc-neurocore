# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestErrorBudget from former test_wave4.py

"""Focused suite: TestErrorBudget from former test_wave4.py."""

from __future__ import annotations

from wave4_support import *  # noqa: F403

class TestErrorBudget:
    def test_no_violation(self):
        eb = ErrorBudget(min_precision=0.90, max_correlation=0.20)
        assert not eb.check(SpikeEvent(precision=0.95, correlation=0.10))

    def test_precision_violation(self):
        eb = ErrorBudget(min_precision=0.90)
        assert eb.check(SpikeEvent(precision=0.85))
        assert eb.violations == 1

    def test_correlation_violation(self):
        eb = ErrorBudget(max_correlation=0.10)
        assert eb.check(SpikeEvent(correlation=0.15))
