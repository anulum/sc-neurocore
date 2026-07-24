# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHeterogeneousParamNames from former test_folded_heterogeneous_params.py

"""Focused suite: TestHeterogeneousParamNames from former test_folded_heterogeneous_params.py."""

from __future__ import annotations

from tests.folded_heterogeneous_params_support import *  # noqa: F403


class TestHeterogeneousParamNames:
    """Which parameters vary per neuron."""

    def test_homogeneous_population_has_no_heterogeneous_params(self) -> None:
        assert _heterogeneous_param_names(_qgraph([10.0, 10.0, 10.0]).populations[0], 16) == []

    def test_varying_tau_is_reported(self) -> None:
        assert _heterogeneous_param_names(_qgraph([10.0, 20.0, 30.0]).populations[0], 16) == ["tau"]

    def test_multiple_varying_parameters_are_sorted(self) -> None:
        pop = _qgraph([10.0, 20.0], v_leak=[0.0, -1.0]).populations[0]
        assert _heterogeneous_param_names(pop, 16) == ["tau", "v_leak"]

    def test_values_that_quantise_equal_are_not_heterogeneous(self) -> None:
        step = 1.0 / (1 << 8)
        pop = _qgraph([10.0, 10.0 + step / 4.0]).populations[0]
        assert _heterogeneous_param_names(pop, 16) == []
