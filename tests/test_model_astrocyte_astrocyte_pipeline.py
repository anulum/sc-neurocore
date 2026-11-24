# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAstrocytePipeline from former test_model_astrocyte.py

"""Focused suite: TestAstrocytePipeline from former test_model_astrocyte.py."""

from __future__ import annotations

from sc_neurocore.network.population import (
    Population,
)
from sc_neurocore.neurons.models.astrocyte import (
    AstrocyteModel,
)


class TestAstrocytePipeline:
    """The astrocyte model in the network pipeline."""

    def test_population_creates(self) -> None:
        """A population of five astrocytes is built."""
        assert Population(AstrocyteModel, n=5, label="astro").n == 5

    def test_returns_float(self) -> None:
        """Rate model (Ca²⁺). Network incompatible (float return)."""
        n = AstrocyteModel()
        assert isinstance(n.step(0.5), float)

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""
        traces = []
        for _ in range(2):
            n = AstrocyteModel()
            trace = [n.step(0.5) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
