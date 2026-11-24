# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBrunelWang from former test_new_models.py

"""Focused suite: TestBrunelWang from former test_new_models.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.brunel_wang import (
    BrunelWangNeuron,
)


class TestBrunelWang:
    """The Brunel-Wang neuron with AMPA, NMDA and GABA inputs."""

    def test_fires_with_ampa(self) -> None:
        """External AMPA drive makes the neuron fire."""
        n = BrunelWangNeuron(dt=0.1)
        # The argument is a summed dimensionless gate, not an injected current.
        spikes = sum(n.step(i_ampa_ext=0.05) for _ in range(5000))
        assert spikes > 0

    def test_subthreshold(self) -> None:
        """Weak drive gives no spikes."""
        n = BrunelWangNeuron(dt=0.1)
        spikes = sum(n.step(i_ampa_ext=0.0) for _ in range(1000))
        assert spikes == 0

    def test_gaba_inhibition(self) -> None:
        """Added GABA inhibition does not raise the spike count."""
        # With strong GABA, should fire less than without
        n1 = BrunelWangNeuron(dt=0.1)
        n2 = BrunelWangNeuron(dt=0.1)
        s1 = sum(n1.step(i_ampa_ext=0.05, s_gaba=0.0) for _ in range(5000))
        s2 = sum(n2.step(i_ampa_ext=0.05, s_gaba=0.08) for _ in range(5000))
        assert s1 >= s2

    def test_nmda_voltage_dep(self) -> None:
        """The NMDA magnesium block is weaker when depolarised than at rest."""
        n = BrunelWangNeuron()
        # At rest (-70mV), Mg block is strong
        block_rest = n._nmda_voltage_dep(-70.0)
        # At depolarised (-20mV), Mg block is weak
        block_depol = n._nmda_voltage_dep(-20.0)
        assert block_depol > block_rest

    def test_refractory(self) -> None:
        """Input during the refractory period gives no spike."""
        n = BrunelWangNeuron(dt=0.1, tau_ref=2.0)
        # Drive until spike
        for _ in range(10000):
            if n.step(i_ampa_ext=0.05):
                break
        # Next step should be refractory (no spike)
        assert n.step(i_ampa_ext=0.05) == 0

    def test_reset(self) -> None:
        """A reset returns the voltage to v_rest."""
        n = BrunelWangNeuron()
        for _ in range(100):
            n.step(i_ampa_ext=0.05)
        n.reset()
        assert n.v == n.v_rest
