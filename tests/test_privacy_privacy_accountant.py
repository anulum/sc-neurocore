# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPrivacyAccountant from former test_privacy.py

"""Focused suite: TestPrivacyAccountant from former test_privacy.py."""

from __future__ import annotations

from tests.privacy_support import *  # noqa: F403


class TestPrivacyAccountant:
    def test_init(self):
        pa = PrivacyAccountant(target_epsilon=2.0)
        assert pa.spent_epsilon == 0.0
        assert pa.remaining_epsilon == 2.0
        assert not pa.budget_exhausted

    def test_record_steps(self):
        pa = PrivacyAccountant(target_epsilon=1.0)
        pa.record_step(0.3)
        pa.record_step(0.3)
        assert pa.spent_epsilon == 0.6
        assert pa.remaining_epsilon == 0.4
        assert not pa.budget_exhausted

    def test_budget_exhausted(self):
        pa = PrivacyAccountant(target_epsilon=0.5)
        pa.record_step(0.6)
        assert pa.budget_exhausted

    def test_summary(self):
        pa = PrivacyAccountant(target_epsilon=1.0)
        pa.record_step(0.1)
        s = pa.summary()
        assert "epsilon" in s
