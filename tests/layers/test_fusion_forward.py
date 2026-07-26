# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC fusion forward tests

"""Weighted modalities, ignored inputs, output shape, and dtype contracts."""

import numpy as np

from sc_neurocore.layers.fusion import SCFusionLayer


def test_fusion_forward_two_modalities() -> None:
    """Fusion output matches weighted sum of inputs."""
    layer = SCFusionLayer(input_dims={"a": 2, "b": 2}, fusion_weights={"a": 1.0, "b": 1.0})
    out = layer.forward({"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])})
    assert np.allclose(out, np.array([0.5, 0.5]))


def test_fusion_forward_single_modality() -> None:
    """Single-modality fusion returns the input scaled by weight."""
    layer = SCFusionLayer(input_dims={"a": 3}, fusion_weights={"a": 2.0})
    out = layer.forward({"a": np.array([0.2, 0.4, 0.6])})
    assert np.allclose(out, np.array([0.2, 0.4, 0.6]))


def test_fusion_ignores_unweighted_modality() -> None:
    """Modalities not in fusion_weights are skipped."""
    layer = SCFusionLayer(input_dims={"a": 2, "b": 2}, fusion_weights={"a": 1.0})
    out = layer.forward({"a": np.array([1.0, 1.0]), "b": np.array([10.0, 10.0])})
    assert np.allclose(out, np.array([1.0, 1.0]))


def test_fusion_missing_weight_modality_skipped() -> None:
    """Missing weights should not affect the fused output."""
    layer = SCFusionLayer(input_dims={"a": 2, "b": 2}, fusion_weights={"b": 1.0})
    out = layer.forward({"a": np.array([5.0, 5.0]), "b": np.array([1.0, 1.0])})
    assert np.allclose(out, np.array([1.0, 1.0]))


def test_fusion_output_shape_matches_input() -> None:
    """Output shape should match input feature dimension."""
    layer = SCFusionLayer(input_dims={"a": 4, "b": 4}, fusion_weights={"a": 1.0, "b": 1.0})
    out = layer.forward({"a": np.ones(4), "b": np.zeros(4)})
    assert out.shape == (4,)


def test_fusion_output_dtype_float() -> None:
    """Output should be float array."""
    layer = SCFusionLayer(input_dims={"a": 2}, fusion_weights={"a": 1.0})
    out = layer.forward({"a": np.array([1, 2])})
    assert np.issubdtype(out.dtype, np.floating)
