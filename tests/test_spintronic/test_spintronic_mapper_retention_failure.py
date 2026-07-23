# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRetentionFailure from former test_spintronic_mapper.py

"""Focused suite: TestRetentionFailure from former test_spintronic_mapper.py."""

from __future__ import annotations

from spintronic_mapper_support import *  # noqa: F403

class TestRetentionFailure:
    def test_high_stability_no_fail(self):
        assert retention_failure_probability(101.0, 3.15e8) == 0.0  # Δ>100 → 0

    def test_low_stability_fails(self):
        p = retention_failure_probability(10.0, 1.0)
        assert p > 0.0
