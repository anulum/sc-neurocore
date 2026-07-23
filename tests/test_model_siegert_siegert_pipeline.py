# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSiegertPipeline from former test_model_siegert.py

"""Focused suite: TestSiegertPipeline from former test_model_siegert.py."""

from __future__ import annotations

from tests.model_siegert_support import *  # noqa: F403

class TestSiegertPipeline:
    def test_population_creates(self) -> None:
        assert Population(SiegertTransferFunction, n=5, label="sieg").n == 5

    def test_returns_float_not_spike(self) -> None:
        """Mean-field rate model. Returns Hz, not binary spike."""
        n = SiegertTransferFunction()
        result = n.step(20.0)
        assert isinstance(result, (float, np.floating))

    def test_deterministic(self) -> None:
        n1 = SiegertTransferFunction()
        n2 = SiegertTransferFunction()
        assert n1.step(20.0) == n2.step(20.0)
