# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDPCertificate from former test_federated_sc.py

"""Focused suite: TestDPCertificate from former test_federated_sc.py."""

from __future__ import annotations

from federated_sc_support import *  # noqa: F403

class TestDPCertificate:
    def test_from_accountant(self):
        acc = PrivacyAccountant(target_epsilon=10.0)
        dp = DPMechanism(epsilon=1.0)
        acc.consume_round(dp, 128)
        cert = DPCertificate.from_accountant(acc, dp, 128)
        assert cert.mechanism == "bitstream_flip_rr"
        assert cert.rounds == 1
        assert cert.delta == 1e-5

    def test_to_dict(self):
        acc = PrivacyAccountant(target_epsilon=10.0)
        dp = DPMechanism(epsilon=1.0)
        acc.consume_round(dp, 128)
        cert = DPCertificate.from_accountant(acc, dp, 128)
        d = cert.to_dict()
        assert "mechanism" in d
        assert "compliant" in d
        assert d["composition_method"] == "renyi_dp"

    def test_compliant_status(self):
        acc = PrivacyAccountant(target_epsilon=100.0)
        dp = DPMechanism(epsilon=1.0)
        acc.consume_round(dp, 64)
        cert = DPCertificate.from_accountant(acc, dp, 64)
        assert cert.is_compliant

    def test_non_compliant_status(self):
        acc = PrivacyAccountant(target_epsilon=0.001)
        dp = DPMechanism(epsilon=1.0)
        for _ in range(100):
            acc.consume_round(dp, 256)
        cert = DPCertificate.from_accountant(acc, dp, 256)
        assert not cert.is_compliant
