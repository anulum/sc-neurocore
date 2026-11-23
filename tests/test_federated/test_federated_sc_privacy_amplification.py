# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPrivacyAmplification from former test_federated_sc.py

"""Focused suite: TestPrivacyAmplification from former test_federated_sc.py."""

from __future__ import annotations

from sc_neurocore.federated.federated_sc import (
    amplified_epsilon,
)


class TestPrivacyAmplification:
    """Privacy amplification by subsampling."""

    def test_full_sampling_no_amplification(self) -> None:
        """Sampling everyone leaves epsilon unchanged."""
        assert amplified_epsilon(1.0, 1.0) == 1.0

    def test_subsampling_reduces_epsilon(self) -> None:
        """Subsampling lowers epsilon."""
        amp = amplified_epsilon(1.0, 0.1)
        assert amp < 1.0

    def test_zero_sampling(self) -> None:
        """Sampling no one gives an epsilon of 0."""
        assert amplified_epsilon(1.0, 0.0) == 0.0

    def test_monotonic_in_rate(self) -> None:
        """Epsilon grows with the sampling rate."""
        a = amplified_epsilon(2.0, 0.1)
        b = amplified_epsilon(2.0, 0.5)
        c = amplified_epsilon(2.0, 1.0)
        assert a < b < c
