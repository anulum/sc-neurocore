# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVectorizedLayerGPU from former test_gpu_backend.py

"""Focused suite: TestVectorizedLayerGPU from former test_gpu_backend.py."""

from __future__ import annotations

from tests.gpu_backend_support import *  # noqa: F403

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
