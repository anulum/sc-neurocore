# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPrivacyAmplification from former test_federated_sc.py

"""Focused suite: TestPrivacyAmplification from former test_federated_sc.py."""

from __future__ import annotations

from federated_sc_support import *  # noqa: F403

class TestPrivacyAmplification:
    def test_full_sampling_no_amplification(self):
        assert amplified_epsilon(1.0, 1.0) == 1.0

    def test_subsampling_reduces_epsilon(self):
        amp = amplified_epsilon(1.0, 0.1)
        assert amp < 1.0

    def test_zero_sampling(self):
        assert amplified_epsilon(1.0, 0.0) == 0.0

    def test_monotonic_in_rate(self):
        a = amplified_epsilon(2.0, 0.1)
        b = amplified_epsilon(2.0, 0.5)
        c = amplified_epsilon(2.0, 1.0)
        assert a < b < c
