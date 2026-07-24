# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (rng_recorder_encoder) from former test_stochastic_core_properties.py

from __future__ import annotations

from tests.stochastic_core_properties_support import *  # noqa: F403


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


@given(n_steps=st.integers(min_value=1, max_value=100))
@settings(max_examples=20)
def test_recorder_accumulates(n_steps):
    rec = BitstreamSpikeRecorder()
    for i in range(n_steps):
        rec.record(i % 2)
    arr = rec.as_array()
    assert len(arr) == n_steps


@given(p=st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=30)
def test_encoder_output_binary(p):
    enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=256)
    bs = enc.encode(p)
    assert set(np.unique(bs)).issubset({0, 1})
