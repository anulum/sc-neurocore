# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVectorizedLayerForward from former test_jax_adapter_fallback_contracts.py

"""Focused suite: TestVectorizedLayerForward from former test_jax_adapter_fallback_contracts.py."""

from __future__ import annotations

from tests.jax_adapter_fallback_contracts_support import *  # noqa: F403


class TestVectorizedLayerForward:
    def test_forward_correct_shape(self):
        from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer

        layer = VectorizedSCLayer(n_inputs=4, n_neurons=8, use_gpu=False)
        result = layer.forward([0.5, 0.5, 0.5, 0.5])
        assert result.shape == (8,)
