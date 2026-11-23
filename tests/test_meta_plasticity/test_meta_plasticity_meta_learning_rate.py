# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMetaLearningRate from former test_meta_plasticity.py

"""Focused suite: TestMetaLearningRate from former test_meta_plasticity.py."""

from __future__ import annotations

from sc_neurocore.meta_plasticity.meta_plasticity import (
    MetaLearningRate,
)


class TestMetaLearningRate:
    """The meta learning rate of the rule evolver."""

    def test_positive_delta_increases(self) -> None:
        """Improving fitness raises the rate."""
        mlr = MetaLearningRate(meta_lr=0.01)
        new = mlr.update(0.1)
        assert new > 0.01

    def test_negative_delta_decreases(self) -> None:
        """Worsening fitness lowers the rate."""
        mlr = MetaLearningRate(meta_lr=0.01)
        new = mlr.update(-0.1)
        assert new < 0.01

    def test_bounded(self) -> None:
        """The rate stays at or below its maximum."""
        mlr = MetaLearningRate(meta_lr=0.01, max_meta_lr=0.1)
        for _ in range(100):
            mlr.update(1.0)
        assert mlr.meta_lr <= 0.1
