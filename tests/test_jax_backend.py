# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for JAX Backend

import pytest
import numpy as np

pytest.importorskip("jax")

from sc_neurocore.accel.jax_backend import (
    jax_forward_pass,
    jax_lif_step,
    jax_pack_bitstream,
    jax_popcount,
    jax_vec_and,
    jax_vec_mac,
    to_host,
)


def test_jax_pack_bitstream_1d():
    bits = np.array([1, 0, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
    packed = jax_pack_bitstream(bits)
    packed_np = to_host(packed)

    assert packed_np.shape == (1,)
    # 1 + 4 + 8 = 13
    assert packed_np[0] == 13


def test_jax_pack_bitstream_2d_binary_matrix():
    bits = np.array([[1, 0, 1, 1], [0, 1, 0, 1]], dtype=np.uint8)
    packed = jax_pack_bitstream(bits)
    packed_np = to_host(packed)

    assert packed_np.shape == (2, 1)
    assert packed_np[0, 0] == 13
    assert packed_np[1, 0] == 10


@pytest.mark.parametrize(
    "bits",
    [
        np.array([1, 0, 2, 1], dtype=np.uint8),
        np.array([1.0, 0.0, 0.25, 1.0], dtype=np.float32),
        np.array([True, False, True, False], dtype=np.bool_),
    ],
)
def test_jax_pack_bitstream_rejects_non_uint8_binary_streams(bits):
    with pytest.raises(ValueError, match="uint8 binary"):
        jax_pack_bitstream(bits)


def test_jax_pack_bitstream_rejects_empty_streams():
    with pytest.raises(ValueError, match="empty"):
        jax_pack_bitstream(np.array([], dtype=np.uint8))


def test_jax_popcount():
    packed = np.array([13, 255, 0], dtype=np.uint64)
    counts = jax_popcount(packed)
    counts_np = to_host(counts)

    assert counts_np[0] == 3
    assert counts_np[1] == 8
    assert counts_np[2] == 0


@pytest.mark.parametrize(
    "packed",
    [
        np.array([13, 255], dtype=np.int64),
        np.array([13.0, 255.0], dtype=np.float64),
        np.array([], dtype=np.uint64),
    ],
)
def test_jax_popcount_rejects_non_uint64_or_empty_inputs(packed):
    with pytest.raises(ValueError, match="uint64"):
        jax_popcount(packed)


def test_jax_vec_and_validates_and_preserves_shape():
    a = np.array([[0b1111, 0b0011], [0b1010, 0b0101]], dtype=np.uint64)
    b = np.array([[0b1100, 0b0101], [0b0011, 0b1111]], dtype=np.uint64)

    result = to_host(jax_vec_and(a, b))

    assert result.shape == a.shape
    assert np.array_equal(result, np.bitwise_and(a, b))


@pytest.mark.parametrize(
    ("a", "b", "match"),
    [
        (
            np.array([1, 3], dtype=np.uint64),
            np.array([1, 3], dtype=np.int64),
            "uint64",
        ),
        (
            np.array([], dtype=np.uint64),
            np.array([], dtype=np.uint64),
            "non-empty",
        ),
        (
            np.array([1, 3], dtype=np.uint64),
            np.array([[1, 3]], dtype=np.uint64),
            "shape",
        ),
    ],
)
def test_jax_vec_and_rejects_invalid_contracts(a, b, match):
    with pytest.raises(ValueError, match=match):
        jax_vec_and(a, b)


def test_jax_vec_mac():
    # 2 neurons, 3 inputs, 1 word each
    weights = np.array([[[3], [1], [0]], [[255], [0], [7]]], dtype=np.uint64)

    inputs = np.array([[3], [3], [3]], dtype=np.uint64)

    # weights & inputs
    # Neuron 0:
    # w0 & in0: 3 & 3 = 3 (pop=2)
    # w1 & in1: 1 & 3 = 1 (pop=1)
    # w2 & in2: 0 & 3 = 0 (pop=0)
    # sum = 3

    # Neuron 1:
    # w0 & in0: 255 & 3 = 3 (pop=2)
    # w1 & in1: 0 & 3 = 0 (pop=0)
    # w2 & in2: 7 & 3 = 3 (pop=2)
    # sum = 4

    out = jax_vec_mac(weights, inputs)
    out_np = to_host(out)

    assert out_np.shape == (2,)
    assert out_np[0] == 3
    assert out_np[1] == 4


@pytest.mark.parametrize(
    ("weights", "inputs", "match"),
    [
        (
            np.array([[3], [1]], dtype=np.uint64),
            np.array([[3], [3]], dtype=np.uint64),
            "3-D",
        ),
        (
            np.array([[[3], [1]]], dtype=np.uint64),
            np.array([3, 3], dtype=np.uint64),
            "2-D",
        ),
        (
            np.array([[[3], [1], [0]]], dtype=np.uint64),
            np.array([[3], [3]], dtype=np.uint64),
            "input dimension",
        ),
        (
            np.array([[[3], [1]]], dtype=np.float64),
            np.array([[3], [3]], dtype=np.uint64),
            "uint64",
        ),
    ],
)
def test_jax_vec_mac_rejects_invalid_contracts(weights, inputs, match):
    with pytest.raises(ValueError, match=match):
        jax_vec_mac(weights, inputs)


def test_jax_lif_step_updates_voltage_and_spikes():
    v = np.array([0.0, 0.9], dtype=np.float64)
    current = np.array([0.4, 0.8], dtype=np.float64)
    noise = np.array([0.0, 0.0], dtype=np.float64)

    v_next, spikes = jax_lif_step(
        v,
        current,
        v_rest=0.0,
        v_reset=-0.1,
        v_threshold=1.0,
        alpha=0.5,
        resistance=1.0,
        noise=noise,
    )

    assert np.allclose(to_host(v_next), np.array([0.4, -0.1]))
    assert np.array_equal(to_host(spikes), np.array([0, 1], dtype=np.uint8))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"resistance": np.inf}, "resistance"),
        ({"v_threshold": np.nan}, "v_threshold"),
    ],
)
def test_jax_lif_step_rejects_invalid_scalar_parameters(kwargs, match):
    v = np.array([0.0, 0.1], dtype=np.float64)
    current = np.array([0.2, 0.3], dtype=np.float64)
    noise = np.zeros(2, dtype=np.float64)
    params = {
        "v_rest": 0.0,
        "v_reset": 0.0,
        "v_threshold": 1.0,
        "alpha": 0.5,
        "resistance": 1.0,
        "noise": noise,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        jax_lif_step(v, current, **params)


@pytest.mark.parametrize(
    ("v", "current", "noise", "match"),
    [
        (
            np.array([0, 1], dtype=np.int64),
            np.array([0.2, 0.3], dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            "floating-point",
        ),
        (
            np.array([0.0, 0.1], dtype=np.float64),
            np.array([[0.2, 0.3]], dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            "shape",
        ),
        (
            np.array([0.0, np.nan], dtype=np.float64),
            np.array([0.2, 0.3], dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            "finite",
        ),
    ],
)
def test_jax_lif_step_rejects_invalid_array_contracts(v, current, noise, match):
    with pytest.raises(ValueError, match=match):
        jax_lif_step(
            v,
            current,
            v_rest=0.0,
            v_reset=0.0,
            v_threshold=1.0,
            alpha=0.5,
            resistance=1.0,
            noise=noise,
        )


def test_jax_forward_pass_returns_layer_spikes_and_final_voltage():
    x = np.array([[0.5, 0.25], [0.0, 1.0]], dtype=np.float64)
    weights = [
        np.array([[0.6, 0.1], [0.2, 0.4]], dtype=np.float64),
        np.array([[0.5, 0.3]], dtype=np.float64),
    ]

    all_spikes, final_v = jax_forward_pass(weights, x, n_steps=3)

    assert len(all_spikes) == 2
    assert to_host(all_spikes[0]).shape == (3, 2, 2)
    assert to_host(all_spikes[1]).shape == (3, 2, 1)
    assert to_host(final_v).shape == (2, 1)


@pytest.mark.parametrize(
    ("weights", "x", "kwargs", "match"),
    [
        ([], np.ones((1, 2), dtype=np.float64), {}, "weights"),
        ([np.ones((1, 2), dtype=np.float64)], np.ones((1, 2), dtype=np.float64), {"n_steps": 0}, "n_steps"),
        ([np.ones((1, 2), dtype=np.float64)], np.ones(2, dtype=np.float64), {}, "2-D"),
        ([np.ones((1, 3), dtype=np.float64)], np.ones((1, 2), dtype=np.float64), {}, "input dimension"),
        ([np.ones((1, 2), dtype=np.float64)], np.array([[np.nan, 0.0]], dtype=np.float64), {}, "finite"),
        ([np.ones((1, 2), dtype=np.float64)], np.ones((1, 2), dtype=np.float64), {"alpha": 0.0}, "alpha"),
    ],
)
def test_jax_forward_pass_rejects_invalid_contracts(weights, x, kwargs, match):
    params = {
        "n_steps": 2,
        "v_rest": 0.0,
        "v_reset": 0.0,
        "v_threshold": 1.0,
        "alpha": 0.9,
    }
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        jax_forward_pass(weights, x, **params)
