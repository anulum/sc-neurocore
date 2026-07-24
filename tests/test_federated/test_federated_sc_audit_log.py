# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAuditLog from former test_federated_sc.py

"""Focused suite: TestAuditLog from former test_federated_sc.py."""

from __future__ import annotations

from federated_sc_support import *  # noqa: F403


class TestAuditLog:
    def test_empty_log(self):
        log = AuditLog()
        assert log.total_rounds == 0
        assert log.max_epsilon == 0.0

    def test_log_round(self):
        log = AuditLog()
        log.log_round(round_number=1, num_active=5, epsilon_consumed=0.5, grad_norm=0.01)
        assert log.total_rounds == 1

    def test_to_list(self):
        log = AuditLog()
        log.log_round(round_number=1, num_active=5, epsilon_consumed=0.5, grad_norm=0.01)
        log.log_round(round_number=2, num_active=3, epsilon_consumed=1.0, grad_norm=0.02)
        entries = log.to_list()
        assert len(entries) == 2
        assert entries[0]["round"] == 1
        assert entries[1]["epsilon"] == 1.0

    def test_max_epsilon(self):
        log = AuditLog()
        log.log_round(1, 5, 0.5, 0.01)
        log.log_round(2, 5, 1.5, 0.01)
        log.log_round(3, 5, 1.0, 0.01)
        assert log.max_epsilon == 1.5
