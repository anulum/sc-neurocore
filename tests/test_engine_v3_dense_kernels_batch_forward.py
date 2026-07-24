# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBatchForward from former test_engine_v3_dense_kernels.py

"""Focused suite: TestBatchForward from former test_engine_v3_dense_kernels.py."""

from __future__ import annotations

from tests.engine_v3_dense_kernels_support import *  # noqa: F403


class TestBatchForward:
    """Verify batched forward API correctness, shape and determinism."""

    def test_batch_vs_sequential(self) -> None:
        layer = v3.DenseLayer(8, 4, 1024, seed=42)
        inputs = np.random.RandomState(42).uniform(0, 1, (10, 8)).astype(np.float64)
        seed = 555

        batched = np.asarray(layer.forward_batch_numpy(inputs, seed=seed), dtype=np.float64)

        sequential_rows = []
        for sample_idx, row in enumerate(inputs):
            sample_seed = seed + sample_idx * 1_000_000
            sequential_rows.append(layer.forward_fast(row.tolist(), seed=sample_seed))
        sequential = np.asarray(sequential_rows, dtype=np.float64)

        np.testing.assert_array_equal(batched, sequential)

    def test_batch_shape(self) -> None:
        layer = v3.DenseLayer(16, 8, 512, seed=42)
        inputs = np.random.RandomState(1).uniform(0, 1, (25, 16)).astype(np.float64)
        out = np.asarray(layer.forward_batch_numpy(inputs, seed=100))
        assert out.shape == (25, 8)

    def test_batch_determinism(self) -> None:
        layer = v3.DenseLayer(16, 8, 512, seed=42)
        inputs = np.random.RandomState(7).uniform(0, 1, (12, 16)).astype(np.float64)
        a = np.asarray(layer.forward_batch_numpy(inputs, seed=101))
        b = np.asarray(layer.forward_batch_numpy(inputs, seed=101))
        np.testing.assert_array_equal(a, b)

    def test_batch_numpy_output(self) -> None:
        layer = v3.DenseLayer(4, 2, 256, seed=42)
        inputs = np.random.RandomState(9).uniform(0, 1, (3, 4)).astype(np.float64)
        out = layer.forward_batch_numpy(inputs)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
