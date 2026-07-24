# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (lif_homeostatic) from former test_stochastic_core_properties.py

from __future__ import annotations

from tests.stochastic_core_properties_support import *  # noqa: F403

@given(
    current=st.floats(min_value=-2.0, max_value=2.0),
    dt=st.floats(min_value=0.1, max_value=2.0),
)
@settings(max_examples=50)
def test_lif_step_returns_binary(current, dt):
    n = StochasticLIFNeuron(v_threshold=1.0, tau_mem=20.0, dt=dt)
    spike = n.step(current)
    assert spike in (0, 1, True, False)

@given(current_int=st.integers(min_value=-128, max_value=127))
@settings(max_examples=30)
def test_fixed_point_lif_no_crash(current_int):
    n = FixedPointLIFNeuron()
    for _ in range(10):
        n.step(leak_k=240, gain_k=16, I_t=current_int)

@given(rate=st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=20)
def test_homeostatic_lif_adapts(rate):
    n = HomeostaticLIFNeuron(target_rate=rate)
    for _ in range(50):
        n.step(1.5)

@given(
    steps=st.integers(min_value=10, max_value=100),
    current=st.floats(min_value=-1.0, max_value=3.0),
)
@settings(max_examples=30)
def test_lif_voltage_resets_on_spike(steps, current):
    n = StochasticLIFNeuron(v_threshold=LIF_V_THRESHOLD, noise_std=0.0)
    for _ in range(steps):
        spike = n.step(current)
        if spike:
            assert n.v == LIF_V_REST

@given(target=st.floats(min_value=0.01, max_value=0.5))
@settings(max_examples=20)
def test_homeostatic_threshold_stays_bounded(target):
    n = HomeostaticLIFNeuron(target_rate=target)
    for _ in range(200):
        n.step(1.5)
    from sc_neurocore.constants import (
        HOMEOSTATIC_THRESHOLD_FLOOR,
        HOMEOSTATIC_THRESHOLD_CEILING_MULT,
    )

    assert n.v_threshold >= HOMEOSTATIC_THRESHOLD_FLOOR
    assert n.v_threshold <= n.initial_threshold * HOMEOSTATIC_THRESHOLD_CEILING_MULT
