# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — maintained default integrator contracts

from __future__ import annotations

from sc_neurocore.neurons.models.adex import AdExNeuron
from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron
from sc_neurocore.neurons.models.morris_lecar import MorrisLecarNeuron
from tests.neuron_integrator_paths_support import count_spikes


def test_default_integrators_match_maintained_paths() -> None:
    hh_default = HodgkinHuxleyNeuron(dt=0.01)
    hh_baseline = HodgkinHuxleyNeuron(dt=0.01, integrator="baseline_euler")
    adex_default = AdExNeuron(dt=0.1)
    adex_baseline = AdExNeuron(dt=0.1, integrator="baseline_euler")
    ml_default = MorrisLecarNeuron(dt=0.05)
    ml_rk4 = MorrisLecarNeuron(dt=0.05, integrator="rk4")
    fhn_default = FitzHughNagumoNeuron(dt=0.05)
    fhn_baseline = FitzHughNagumoNeuron(dt=0.05, integrator="baseline_euler")

    hh_default_spikes = count_spikes(hh_default, 10.0, 200)
    hh_baseline_spikes = count_spikes(hh_baseline, 10.0, 200)
    adex_default_spikes = count_spikes(adex_default, 500.0, 2000)
    adex_baseline_spikes = count_spikes(adex_baseline, 500.0, 2000)
    ml_default_spikes = count_spikes(ml_default, 100.0, 1000)
    ml_rk4_spikes = count_spikes(ml_rk4, 100.0, 1000)
    fhn_default_spikes = count_spikes(fhn_default, 0.8, 1000)
    fhn_baseline_spikes = count_spikes(fhn_baseline, 0.8, 1000)

    assert hh_default_spikes == hh_baseline_spikes
    assert adex_default_spikes == adex_baseline_spikes
    assert ml_default_spikes == ml_rk4_spikes
    assert fhn_default_spikes == fhn_baseline_spikes
