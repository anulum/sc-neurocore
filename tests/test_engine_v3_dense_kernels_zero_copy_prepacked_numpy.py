# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestZeroCopyPrepackedNumpy from former test_engine_v3_dense_kernels.py

"""Focused suite: TestZeroCopyPrepackedNumpy from former test_engine_v3_dense_kernels.py."""

from __future__ import annotations

from tests.engine_v3_dense_kernels_support import *  # noqa: F403


class TestZeroCopyPrepackedNumpy:
    """Tests for forward_prepacked_numpy (true zero-copy path)."""

    def test_output_shape_and_type(self) -> None:
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        probs = np.array([0.3, 0.5, 0.7, 0.9], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=55)
        out = layer.forward_prepacked_numpy(packed)
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)
        assert out.dtype == np.float64

    def test_matches_forward_prepacked(self) -> None:
        """Zero-copy numpy path must match the existing prepacked path."""
        layer = v3.DenseLayer(8, 4, 512, seed=42)
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=512, seed=99)
        out_legacy = layer.forward_prepacked(packed)
        out_numpy = layer.forward_prepacked_numpy(packed)
        np.testing.assert_allclose(out_numpy, out_legacy)

    def test_wrong_n_inputs(self) -> None:
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        packed = np.zeros((3, 16), dtype=np.uint64)  # 3 inputs, need 4
        with pytest.raises(ValueError):
            layer.forward_prepacked_numpy(packed)

    def test_wrong_word_count(self) -> None:
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        packed = np.zeros((4, 10), dtype=np.uint64)  # 10 words, need 16
        with pytest.raises(ValueError):
            layer.forward_prepacked_numpy(packed)

    def test_pipeline_encode_then_zero_copy(self) -> None:
        """Full pipeline: batch_encode_numpy -> forward_prepacked_numpy."""
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        probs = np.random.uniform(0, 1, 16)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=42)
        out = layer.forward_prepacked_numpy(packed)
        assert out.shape == (8,)
        assert np.all(out >= 0.0)

    def test_deterministic(self) -> None:
        layer = v3.DenseLayer(4, 2, 512, seed=42)
        probs = np.array([0.5] * 4, dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=512, seed=42)
        out1 = layer.forward_prepacked_numpy(packed)
        out2 = layer.forward_prepacked_numpy(packed)
        np.testing.assert_array_equal(out1, out2)
