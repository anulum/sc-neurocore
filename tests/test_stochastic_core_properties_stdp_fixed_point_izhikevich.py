# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (stdp_fixed_point_izhikevich) from former test_stochastic_core_properties.py

from __future__ import annotations

from tests.stochastic_core_properties_support import *  # noqa: F403


@given(
    w=st.floats(min_value=-1.0, max_value=2.0),
    lr=st.floats(min_value=0.001, max_value=0.1),
)
@settings(max_examples=30)
def test_stdp_weight_stays_bounded(w, lr):
    w_clamped = max(0.0, min(1.0, w))
    syn = StochasticSTDPSynapse(w_min=0.0, w_max=1.0, w=w_clamped, learning_rate=lr, length=64)
    for _ in range(50):
        syn.process_step(pre_bit=1, post_bit=1)
    assert 0.0 <= syn.w <= 1.0


@given(reward=st.floats(min_value=-5.0, max_value=5.0))
@settings(max_examples=30)
def test_rstdp_weight_bounded_after_reward(reward):
    syn = RewardModulatedSTDPSynapse(w_min=0.0, w_max=1.0, w=0.5, length=64)
    for _ in range(20):
        syn.process_step(pre_bit=1, post_bit=1)
    syn.apply_reward(reward)
    assert 0.0 <= syn.w <= 1.0


@given(
    I_t=st.integers(min_value=-(1 << 15), max_value=(1 << 15) - 1),
    leak_k=st.integers(min_value=0, max_value=255),
    gain_k=st.integers(min_value=0, max_value=255),
)
@settings(max_examples=50)
def test_fixed_point_lif_no_overflow(I_t, leak_k, gain_k):
    n = FixedPointLIFNeuron()
    for _ in range(20):
        spike, v = n.step(leak_k=leak_k, gain_k=gain_k, I_t=I_t)
        assert spike in (0, 1)
        W = FP_DATA_WIDTH
        assert -(1 << (W - 1)) <= v < (1 << (W - 1))


@given(seed=st.integers(min_value=1, max_value=0xFFFF))
@settings(max_examples=20)
def test_fixed_point_encoder_output_binary(seed):
    enc = FixedPointBitstreamEncoder(seed_init=seed)
    bits = [enc.step(x_value=128) for _ in range(100)]
    assert all(b in (0, 1) for b in bits)


@given(
    steps=st.integers(min_value=10, max_value=50),
    current=st.floats(min_value=0.0, max_value=10.0),
)
@settings(max_examples=20)
def test_izhikevich_spike_resets_voltage(steps, current):
    n = SCIzhikevichNeuron(noise_std=0.0)
    for _ in range(steps):
        spike = n.step(current)
        if spike:
            assert n.v == n.c
