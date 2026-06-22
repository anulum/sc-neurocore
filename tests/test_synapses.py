# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Synapses

import numpy as np
import pytest
from sc_neurocore.synapses.sc_synapse import BitstreamSynapse
from sc_neurocore.utils.bitstreams import bitstream_to_probability


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("w_min", float("nan")),
        ("w_max", float("inf")),
        ("length", 0),
        ("length", True),
        ("w", float("nan")),
        ("w", -0.1),
        ("w", 1.1),
    ],
)
def test_invalid_synapse_parameters_fail_closed(field, value):
    kwargs = {"w_min": 0.0, "w_max": 1.0, "length": 1024, "w": 0.5, "seed": 42}
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        BitstreamSynapse(**kwargs)


def test_invalid_synapse_weight_bounds_fail_closed():
    with pytest.raises(ValueError, match="w_min"):
        BitstreamSynapse(w_min=1.0, w_max=1.0, length=1024, w=1.0)


def test_synapse_encoding():
    syn = BitstreamSynapse(w_min=0.0, w_max=1.0, length=1000, w=0.5, seed=42)
    assert len(syn.weight_bits) == 1000
    p_eff = syn.effective_weight_probability()
    assert np.isclose(p_eff, 0.5, atol=0.05)


def test_synapse_multiplication():
    # P(A AND B) = P(A) * P(B) if independent
    length = 10000
    syn = BitstreamSynapse(w_min=0.0, w_max=1.0, length=length, w=0.5, seed=42)

    # Create input with p=0.8
    # We use a different seed implicitly or manually to ensure independence
    # BitstreamSynapse uses an internal RNG for encoding.
    # Let's generate an input stream.
    from sc_neurocore.utils.bitstreams import generate_bernoulli_bitstream

    input_bits = generate_bernoulli_bitstream(0.8, length)

    output_bits = syn.apply(input_bits)

    p_out = bitstream_to_probability(output_bits)
    expected = 0.5 * 0.8  # 0.4

    assert np.isclose(p_out, expected, atol=0.05)


def test_synapse_update():
    syn = BitstreamSynapse(w_min=0.0, w_max=1.0, length=1024, w=0.2)
    initial_p = syn.effective_weight_probability()

    syn.update_weight(0.8)
    new_p = syn.effective_weight_probability()

    assert not np.isclose(initial_p, new_p, atol=0.15)
    assert np.isclose(new_p, 0.8, atol=0.15)


@pytest.mark.parametrize("new_w", [float("nan"), -0.1, 1.1])
def test_invalid_weight_update_fails_closed_without_mutation(new_w):
    syn = BitstreamSynapse(w_min=0.0, w_max=1.0, length=1024, w=0.5, seed=42)
    old_w = syn.w
    old_bits = syn.weight_bits.copy()

    with pytest.raises(ValueError, match="w"):
        syn.update_weight(new_w)

    assert syn.w == old_w
    assert np.array_equal(syn.weight_bits, old_bits)


@pytest.mark.parametrize(
    "pre_bits",
    [
        np.array([0, 1, 2, 1], dtype=np.uint8),
        np.ones((4, 1), dtype=np.uint8),
        [0, 1, 0, 1],
    ],
)
def test_invalid_pre_bitstream_fails_closed(pre_bits):
    syn = BitstreamSynapse(w_min=0.0, w_max=1.0, length=4, w=0.5, seed=42)

    with pytest.raises(ValueError, match="pre_bits"):
        syn.apply(pre_bits)


def test_apply_rejects_bitstream_length_mismatch():
    # A correctly-typed 1-D binary bitstream whose length differs from the
    # weight bitstream cannot be multiplied element-wise.
    syn = BitstreamSynapse(w_min=0.0, w_max=1.0, length=256, w=0.5, seed=42)
    pre_bits = np.ones(128, dtype=np.uint8)

    with pytest.raises(ValueError, match="length mismatch"):
        syn.apply(pre_bits)
