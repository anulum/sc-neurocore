# SPDX-License-Identifier: AGPL-3.0-or-later
"""Property-based tests for SC-NeuroCore core modules using Hypothesis."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from sc_neurocore import (
    BitstreamEncoder,
    generate_bernoulli_bitstream,
    bitstream_to_probability,
    StochasticLIFNeuron,
    FixedPointLIFNeuron,
    FixedPointLFSR,
    FixedPointBitstreamEncoder,
    SCDenseLayer,
    VectorizedSCLayer,
    RNG,
    BitstreamSpikeRecorder,
    HomeostaticLIFNeuron,
    SCIzhikevichNeuron,
    StochasticSTDPSynapse,
    RewardModulatedSTDPSynapse,
)
from sc_neurocore.constants import (
    LIF_V_REST,
    LIF_V_THRESHOLD,
    FP_DATA_WIDTH,
)


# -- Bitstream encoding properties --


@given(p=st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=50)
def test_bernoulli_bitstream_values_are_binary(p):
    bs = generate_bernoulli_bitstream(p, length=256)
    assert set(np.unique(bs)).issubset({0, 1})


@given(p=st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=50)
def test_bitstream_roundtrip_within_tolerance(p):
    bs = generate_bernoulli_bitstream(p, length=4096)
    recovered = bitstream_to_probability(bs)
    assert abs(recovered - p) < 0.1


@given(length=st.integers(min_value=1, max_value=2048))
@settings(max_examples=30)
def test_bernoulli_bitstream_correct_length(length):
    bs = generate_bernoulli_bitstream(0.5, length=length)
    assert len(bs) == length


@given(p=st.just(0.0))
def test_zero_probability_all_zeros(p):
    bs = generate_bernoulli_bitstream(p, length=1024)
    assert np.sum(bs) == 0


@given(p=st.just(1.0))
def test_one_probability_all_ones(p):
    bs = generate_bernoulli_bitstream(p, length=1024)
    assert np.sum(bs) == 1024


# -- LFSR properties --


@given(seed=st.integers(min_value=1, max_value=0xFFFF))
@settings(max_examples=30)
def test_lfsr_nonzero_output(seed):
    lfsr = FixedPointLFSR(seed=seed)
    vals = [lfsr.step() for _ in range(100)]
    assert any(v != 0 for v in vals)


@given(seed=st.integers(min_value=1, max_value=0xFFFF))
@settings(max_examples=20)
def test_lfsr_deterministic(seed):
    a = FixedPointLFSR(seed=seed)
    b = FixedPointLFSR(seed=seed)
    assert [a.step() for _ in range(50)] == [b.step() for _ in range(50)]


# -- Neuron properties --


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


@given(rate=st.floats(min_value=0.1, max_value=100.0))
@settings(max_examples=20)
def test_homeostatic_lif_adapts(rate):
    n = HomeostaticLIFNeuron(target_rate=rate)
    for _ in range(50):
        n.step(1.5)


# -- Layer properties --


@given(
    n_in=st.integers(min_value=2, max_value=8),
    n_out=st.integers(min_value=2, max_value=8),
)
@settings(max_examples=20)
def test_dense_layer_produces_spikes(n_in, n_out):
    inputs = [0.5] * n_in
    weights = [0.3] * n_in
    layer = SCDenseLayer(
        n_neurons=n_out,
        x_inputs=inputs,
        weight_values=weights,
        x_min=0.0,
        x_max=1.0,
        w_min=-1.0,
        w_max=1.0,
        length=64,
    )
    layer.run(T=50)
    trains = layer.get_spike_trains()
    assert len(trains) == n_out
    assert all(len(t) == 50 for t in trains)


@given(
    n_in=st.integers(min_value=2, max_value=16),
    n_out=st.integers(min_value=2, max_value=16),
)
@settings(max_examples=20)
def test_vectorized_layer_output_shape(n_in, n_out):
    layer = VectorizedSCLayer(n_inputs=n_in, n_neurons=n_out, length=64)
    inp = np.random.rand(n_in)
    out = layer.forward(inp)
    assert out.shape == (n_out,)


@given(
    n_in=st.integers(min_value=2, max_value=8),
    n_out=st.integers(min_value=2, max_value=8),
)
@settings(max_examples=10)
def test_vectorized_layer_output_bounded(n_in, n_out):
    layer = VectorizedSCLayer(n_inputs=n_in, n_neurons=n_out, length=128)
    inp = np.random.rand(n_in)
    out = layer.forward(inp)
    assert np.all(np.isfinite(out))


# -- RNG properties --


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=20)
def test_rng_deterministic(seed):
    a = RNG(seed=seed)
    b = RNG(seed=seed)
    va = a.random(100)
    vb = b.random(100)
    np.testing.assert_array_equal(va, vb)


@given(n=st.integers(min_value=1, max_value=1000))
@settings(max_examples=20)
def test_rng_output_shape(n):
    r = RNG(seed=42)
    out = r.random(n)
    assert out.shape == (n,)


@given(n=st.integers(min_value=100, max_value=500))
@settings(max_examples=10)
def test_rng_output_range(n):
    r = RNG(seed=42)
    out = r.random(n)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


# -- Recorder properties --


@given(n_steps=st.integers(min_value=1, max_value=100))
@settings(max_examples=20)
def test_recorder_accumulates(n_steps):
    rec = BitstreamSpikeRecorder()
    for i in range(n_steps):
        rec.record(i % 2)
    arr = rec.as_array()
    assert len(arr) == n_steps


# -- Encoder properties --


@given(p=st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=30)
def test_encoder_output_binary(p):
    enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=256)
    bs = enc.encode(p)
    assert set(np.unique(bs)).issubset({0, 1})


# -- Synapse weight clamping properties --


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


# -- Fixed-point overflow properties --


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


# -- Membrane voltage invariant --


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


# -- Homeostatic adaptation invariant --


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
