# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMutualInformation from former test_spike_train_stats.py

"""Focused suite: TestMutualInformation from former test_spike_train_stats.py."""

from __future__ import annotations

from sc_neurocore.analysis.spike_stats import (
    mutual_information,
)
from tests.spike_train_stats_support import (
    _poisson_train,
)


class TestMutualInformation:
    """The mutual information of two binned spike trains."""

    def test_self_positive(self) -> None:
        """A train has positive mutual information with itself."""
        train = _poisson_train(100.0, 1.0)
        mi = mutual_information(train, train, bin_size=20)
        assert mi > 0

    def test_independent_low(self) -> None:
        """Two independent trains share less information than a train shares with itself."""
        a = _poisson_train(50.0, 1.0, seed=1)
        b = _poisson_train(50.0, 1.0, seed=99)
        mi = mutual_information(a, b, bin_size=20)
        mi_self = mutual_information(a, a, bin_size=20)
        assert mi < mi_self
