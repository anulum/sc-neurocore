"""
Tests for sc_neurocore.accel.gpu_backend.

All tests run on both CuPy (GPU) and NumPy (CPU) — the backend is
selected automatically at import time.
"""

import numpy as np
import pytest

from sc_neurocore.accel.gpu_backend import (
    xp,
    HAS_CUPY,
    to_device,
    to_host,
    gpu_pack_bitstream,
    gpu_vec_and,
    gpu_popcount,
    gpu_vec_mac,
)


class TestTransferHelpers:
    def test_to_device_and_back(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        dev = to_device(a)
        host = to_host(dev)
        np.testing.assert_array_equal(host, a)

    def test_to_host_passthrough(self):
        a = np.array([4, 5, 6])
        assert to_host(a) is a or np.array_equal(to_host(a), a)


class TestGPUPack:
    def test_1d_roundtrip(self):
        bits = np.array([1, 0, 1, 1] + [0] * 60, dtype=np.uint8)
        packed = gpu_pack_bitstream(to_device(bits))
        host = to_host(packed)
        assert host.dtype == np.uint64
        assert host.shape == (1,)
        # bit0=1, bit1=0, bit2=1, bit3=1 -> 0b1101 = 13
        assert int(host[0]) == 13

    def test_2d_shape(self):
        bits = xp.zeros((4, 128), dtype=xp.uint8)
        packed = gpu_pack_bitstream(bits)
        assert packed.shape == (4, 2)  # 128/64 = 2 words

    def test_all_ones(self):
        bits = xp.ones(64, dtype=xp.uint8)
        packed = gpu_pack_bitstream(bits)
        assert int(to_host(packed)[0]) == (2**64 - 1)

    def test_all_zeros(self):
        bits = xp.zeros(64, dtype=xp.uint8)
        packed = gpu_pack_bitstream(bits)
        assert int(to_host(packed)[0]) == 0


class TestGPUVecAnd:
    def test_identity(self):
        a = xp.array([0xFFFFFFFFFFFFFFFF], dtype=xp.uint64)
        b = xp.array([0xAAAAAAAAAAAAAAAA], dtype=xp.uint64)
        result = gpu_vec_and(a, b)
        assert int(to_host(result)[0]) == 0xAAAAAAAAAAAAAAAA

    def test_zero(self):
        a = xp.array([0xFFFFFFFFFFFFFFFF], dtype=xp.uint64)
        b = xp.array([0], dtype=xp.uint64)
        result = gpu_vec_and(a, b)
        assert int(to_host(result)[0]) == 0


class TestGPUPopcount:
    def test_known_values(self):
        # 0xFF = 8 bits set, packed in uint64
        packed = xp.array([0xFF, 0xFFFF, 0], dtype=xp.uint64)
        counts = to_host(gpu_popcount(packed))
        assert counts[0] == 8
        assert counts[1] == 16
        assert counts[2] == 0

    def test_all_ones(self):
        packed = xp.array([0xFFFFFFFFFFFFFFFF], dtype=xp.uint64)
        assert int(to_host(gpu_popcount(packed))[0]) == 64

    def test_single_bit(self):
        for bit_pos in [0, 1, 31, 63]:
            packed = xp.array([1 << bit_pos], dtype=xp.uint64)
            assert int(to_host(gpu_popcount(packed))[0]) == 1


class TestGPUVecMAC:
    def test_simple_mac(self):
        # 2 neurons, 1 input, 1 word
        # weights all 1s, input all 1s -> popcount = 64
        w = xp.array([[[0xFFFFFFFFFFFFFFFF]], [[0]]], dtype=xp.uint64)
        inp = xp.array([[0xFFFFFFFFFFFFFFFF]], dtype=xp.uint64)
        result = to_host(gpu_vec_mac(w, inp))
        assert result[0] == 64
        assert result[1] == 0

    def test_multiple_inputs(self):
        # 1 neuron, 2 inputs, 1 word each — all 1s
        w = xp.array([[[0xFFFFFFFFFFFFFFFF], [0xFFFFFFFFFFFFFFFF]]], dtype=xp.uint64)
        inp = xp.array([[0xFFFFFFFFFFFFFFFF], [0xFFFFFFFFFFFFFFFF]], dtype=xp.uint64)
        result = to_host(gpu_vec_mac(w, inp))
        assert result[0] == 128  # 64 + 64

    def test_zero_input(self):
        w = xp.ones((4, 3, 2), dtype=xp.uint64) * 0xFFFFFFFFFFFFFFFF
        inp = xp.zeros((3, 2), dtype=xp.uint64)
        result = to_host(gpu_vec_mac(w, inp))
        np.testing.assert_array_equal(result, 0)


class TestVectorizedLayerGPU:
    """Integration test: VectorizedSCLayer with GPU path."""

    def test_forward_shape(self):
        from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer

        layer = VectorizedSCLayer(n_inputs=4, n_neurons=8, length=256)
        out = layer.forward([0.5, 0.5, 0.5, 0.5])
        assert out.shape == (8,)

    def test_zero_input_low_output(self):
        from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer

        layer = VectorizedSCLayer(n_inputs=4, n_neurons=8, length=1024)
        out = layer.forward([0.0, 0.0, 0.0, 0.0])
        assert np.all(out < 0.05)

    def test_high_input_positive_output(self):
        from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer

        layer = VectorizedSCLayer(n_inputs=4, n_neurons=8, length=1024)
        out = layer.forward([0.9, 0.9, 0.9, 0.9])
        assert np.all(out > 0.1)
