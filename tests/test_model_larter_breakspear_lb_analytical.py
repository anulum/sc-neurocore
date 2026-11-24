# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLBAnalytical from former test_model_larter_breakspear.py

"""Focused suite: TestLBAnalytical from former test_model_larter_breakspear.py."""

from __future__ import annotations

import pytest
from sc_neurocore.neurons.models.larter_breakspear import (
    LarterBreakspearNeuron,
)


class TestLBAnalytical:
    """Analytical behaviour of the LarterBreakspearNeuron."""

    def test_m_ca_tanh(self) -> None:
        """m_Ca = 0.5·(1 + tanh((V+0.01)/0.15))."""
        n = LarterBreakspearNeuron()
        # At V=-0.01: m_Ca = 0.5
        assert abs(n._m_ca(-0.01) - 0.5) < 1e-10

    def test_m_na_tanh(self) -> None:
        """m_Na = 0.5·(1 + tanh((V-T_Na)/delta_Na))."""
        n = LarterBreakspearNeuron()
        assert abs(n._m_na(n.t_na) - 0.5) < 1e-10

    def test_m_k_tanh(self) -> None:
        """m_K = 0.5·(1 + tanh((V-T_K)/delta_K))."""
        n = LarterBreakspearNeuron()
        assert abs(n._m_k(n.t_k) - 0.5) < 1e-10

    def test_four_currents_positive_conductances(self) -> None:
        """The four ionic conductances are positive."""
        n = LarterBreakspearNeuron()
        assert n.g_ca > 0 and n.g_na > 0 and n.g_k > 0 and n.g_l > 0

    def test_source_population_feedback_is_present(self) -> None:
        """Source recurrence carries AMPA/NMDA excitation and inhibition."""
        n = LarterBreakspearNeuron()
        assert n.a_ee > 0 and n.r_nmda > 0 and n.a_ie > 0
        assert n._q_v(n.v_t) == pytest.approx(0.5 * n.q_v_max)  # noqa: F405
        assert n._q_z(n.z_t) == pytest.approx(0.5 * n.q_z_max)  # noqa: F405

    def test_output_is_voltage(self) -> None:
        """step() returns V directly."""
        n = LarterBreakspearNeuron()
        result = n.step(0.0)
        assert result == n.v

    def test_three_state_variables(self) -> None:
        """The model has three state variables."""
        n = LarterBreakspearNeuron()
        for attr in ["v", "w", "z"]:
            assert hasattr(n, attr)
