# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPopulationParamsAreUniform from former test_folded_heterogeneous_params.py

"""Focused suite: TestPopulationParamsAreUniform from former test_folded_heterogeneous_params.py."""

from __future__ import annotations

from tests.folded_heterogeneous_params_support import *  # noqa: F403


class TestPopulationParamsAreUniform:
    """The per-population parameter-uniformity predicate."""

    def test_homogeneous_population_is_uniform(self) -> None:
        assert (
            _population_params_are_uniform(_qgraph([10.0, 10.0, 10.0]).populations[0], 16) is True
        )

    def test_heterogeneous_tau_is_not_uniform(self) -> None:
        assert (
            _population_params_are_uniform(_qgraph([10.0, 20.0, 30.0]).populations[0], 16) is False
        )
