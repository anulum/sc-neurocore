# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestForwardNumpy from former test_engine_v3_dense_kernels.py

"""Focused suite: TestForwardNumpy from former test_engine_v3_dense_kernels.py."""

from __future__ import annotations

from tests.engine_v3_dense_kernels_support import *  # noqa: F403

class TestForwardNumpy:
    """Tests for single-call numpy dense forward."""

    def test_output_shape_and_type(self) -> None:
        layer = v3.DenseLayer(16, 8, 512)
        inputs = np.array([0.5] * 16, dtype=np.float64)
        out = layer.forward_numpy(inputs)
        assert isinstance(out, np.ndarray)
        assert out.shape == (8,)
        assert out.dtype == np.float64

    def test_output_range(self) -> None:
        layer = v3.DenseLayer(16, 8, 512)
        inputs = np.array([0.3] * 16, dtype=np.float64)
        out = layer.forward_numpy(inputs)
        assert np.all(out >= 0.0)
        assert np.all(out <= 16.0)

    def test_deterministic(self) -> None:
        layer = v3.DenseLayer(16, 8, 512, seed=42)
        inputs = np.array([0.5] * 16, dtype=np.float64)
        out1 = layer.forward_numpy(inputs, seed=100)
        out2 = layer.forward_numpy(inputs, seed=100)
        np.testing.assert_array_equal(out1, out2)

    def test_matches_forward_fast(self) -> None:
        """forward_numpy should match forward_fast with same seed."""
        layer = v3.DenseLayer(8, 4, 256, seed=42)
        inputs_list = [0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8]
        inputs_np = np.array(inputs_list, dtype=np.float64)
        out_fast = layer.forward_fast(inputs_list, seed=42)
        out_numpy = layer.forward_numpy(inputs_np, seed=42)
        np.testing.assert_allclose(out_numpy, out_fast)

    def test_wrong_input_length(self) -> None:
        layer = v3.DenseLayer(8, 4, 256)
        inputs = np.array([0.5] * 7, dtype=np.float64)
        with pytest.raises(ValueError):
            layer.forward_numpy(inputs)

    def test_different_seed_different_output(self) -> None:
        layer = v3.DenseLayer(8, 4, 1024, seed=42)
        inputs = np.array([0.5] * 8, dtype=np.float64)
        out1 = layer.forward_numpy(inputs, seed=100)
        out2 = layer.forward_numpy(inputs, seed=200)
        assert not np.array_equal(out1, out2)
