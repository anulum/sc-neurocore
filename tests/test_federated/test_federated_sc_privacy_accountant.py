# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPrivacyAccountant from former test_federated_sc.py

"""Focused suite: TestPrivacyAccountant from former test_federated_sc.py."""

from __future__ import annotations

from federated_sc_support import *  # noqa: F403

class TestPrivacyAccountant:
    def test_initial_state(self):
        acc = PrivacyAccountant(target_epsilon=10.0)
        assert acc.current_epsilon() == 0.0
        assert not acc.is_exhausted()
        assert acc.rounds_consumed == 0

    def test_consume_round(self):
        acc = PrivacyAccountant(target_epsilon=100.0)
        dp = DPMechanism(epsilon=1.0)
        result = acc.consume_round(dp, 64)
        assert result is True
        assert acc.rounds_consumed == 1
        assert acc.current_epsilon() > 0

    def test_budget_exhaustion(self):
        acc = PrivacyAccountant(target_epsilon=0.01, target_delta=1e-5)
        dp = DPMechanism(epsilon=1.0)
        for _ in range(100):
            acc.consume_round(dp, 256)
        assert acc.is_exhausted()

    def test_remaining_epsilon(self):
        acc = PrivacyAccountant(target_epsilon=100.0)
        assert acc.remaining_epsilon() == 100.0
        dp = DPMechanism(epsilon=1.0)
        acc.consume_round(dp, 64)
        assert acc.remaining_epsilon() < 100.0
