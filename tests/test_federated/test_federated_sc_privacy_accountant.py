# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPrivacyAccountant from former test_federated_sc.py

"""Focused suite: TestPrivacyAccountant from former test_federated_sc.py."""

from __future__ import annotations

from sc_neurocore.federated.federated_sc import (
    DPMechanism,
    PrivacyAccountant,
)


class TestPrivacyAccountant:
    """Accounting the privacy budget over rounds."""

    def test_initial_state(self) -> None:
        """A new accountant has spent nothing."""
        acc = PrivacyAccountant(target_epsilon=10.0)
        assert acc.current_epsilon() == 0.0
        assert not acc.is_exhausted()
        assert acc.rounds_consumed == 0

    def test_consume_round(self) -> None:
        """A consumed round is counted and spends epsilon."""
        acc = PrivacyAccountant(target_epsilon=100.0)
        dp = DPMechanism(epsilon=1.0)
        result = acc.consume_round(dp, 64)
        assert result is True
        assert acc.rounds_consumed == 1
        assert acc.current_epsilon() > 0

    def test_budget_exhaustion(self) -> None:
        """Enough rounds exhaust the budget."""
        acc = PrivacyAccountant(target_epsilon=0.01, target_delta=1e-5)
        dp = DPMechanism(epsilon=1.0)
        for _ in range(100):
            acc.consume_round(dp, 256)
        assert acc.is_exhausted()

    def test_remaining_epsilon(self) -> None:
        """The remaining epsilon falls from the full budget."""
        acc = PrivacyAccountant(target_epsilon=100.0)
        assert acc.remaining_epsilon() == 100.0
        dp = DPMechanism(epsilon=1.0)
        acc.consume_round(dp, 64)
        assert acc.remaining_epsilon() < 100.0
