# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestScKGBERTInterface from former test_transcriptomic.py

"""Focused suite: TestScKGBERTInterface from former test_transcriptomic.py."""

from __future__ import annotations

from tests.transcriptomic_support import *  # noqa: F403

class TestScKGBERTInterface:
    def test_defaults(self) -> None:
        iface = ScKGBERTInterface(n_genes=50)
        assert iface.d_model == 64
        assert iface.sigma == pytest.approx(1.0)

    def test_gaussian_attention_shape(self) -> None:
        iface = ScKGBERTInterface(d_model=8, n_genes=20)
        q = np.random.default_rng(1).normal(0, 1, (5, 8))
        k = np.random.default_rng(2).normal(0, 1, (7, 8))
        v = np.random.default_rng(3).normal(0, 1, (7, 8))
        out = iface.gaussian_attention(q, k, v)
        assert out.shape == (5, 8)

    def test_gaussian_attention_weights_sum_to_one(self) -> None:
        """Gaussian kernel weights must form proper distribution."""
        iface = ScKGBERTInterface(d_model=4, n_genes=10, sigma=1.0)
        q = np.array([[1.0, 0.0, 0.0, 0.0]])
        k = np.random.default_rng(5).normal(0, 1, (6, 4))
        v = np.ones((6, 4))
        out = iface.gaussian_attention(q, k, v)
        # If all values are 1 and weights sum to 1 → output ≈ 1
        np.testing.assert_allclose(out[0], 1.0, atol=1e-6)

    def test_gaussian_attention_concentrates_on_nearest(self) -> None:
        """Small sigma → attention concentrates on nearest key."""
        iface = ScKGBERTInterface(d_model=2, n_genes=10, sigma=0.01)
        q = np.array([[0.0, 0.0]])
        k = np.array([[0.0, 0.0], [10.0, 10.0]])
        v = np.array([[1.0, 0.0], [0.0, 1.0]])
        out = iface.gaussian_attention(q, k, v)
        # Should almost entirely attend to first key (distance=0)
        np.testing.assert_allclose(out[0], [1.0, 0.0], atol=1e-4)

    def test_encode_expression_shape(self) -> None:
        iface = ScKGBERTInterface(d_model=16, n_genes=100)
        expr = np.random.default_rng(1).poisson(3, 100).astype(np.float64)
        emb = iface.encode_expression(expr)
        assert emb.shape == (16,)

    def test_encode_all_zeros(self) -> None:
        iface = ScKGBERTInterface(d_model=8, n_genes=20)
        emb = iface.encode_expression(np.zeros(20))
        assert np.allclose(emb, 0.0)

    def test_dual_encoder_differs_from_single(self) -> None:
        """encode_with_knowledge incorporates KG → different from encode_expression."""
        iface = ScKGBERTInterface(d_model=16, n_genes=50, seed=42)
        expr = np.random.default_rng(7).poisson(2, 50).astype(np.float64)
        s_emb = iface.encode_expression(expr)
        k_emb = iface.encode_with_knowledge(expr)
        assert not np.allclose(s_emb, k_emb)

    def test_predict_cell_type(self) -> None:
        iface = ScKGBERTInterface(d_model=8, n_genes=30, seed=1)
        rng = np.random.default_rng(10)
        # Create two distinct prototype profiles
        proto_a = rng.poisson(5, 30).astype(np.float64)
        proto_b = rng.poisson(1, 30).astype(np.float64)
        emb_a = iface.encode_with_knowledge(proto_a)
        emb_b = iface.encode_with_knowledge(proto_b)
        prototypes = np.array([emb_a, emb_b])
        labels = ["neuron", "glia"]
        # Query close to proto_a
        pred = iface.predict_cell_type(proto_a, prototypes, labels)
        assert pred == "neuron"

    def test_gene_importance_nonzero(self) -> None:
        iface = ScKGBERTInterface(d_model=8, n_genes=30, seed=3)
        expr = np.random.default_rng(5).poisson(3, 30).astype(np.float64)
        imp = iface.gene_importance(expr)
        assert imp.shape == (30,)
        assert imp.sum() > 0

    def test_gene_importance_zeros_for_unexpressed(self) -> None:
        iface = ScKGBERTInterface(d_model=8, n_genes=20, seed=1)
        expr = np.zeros(20)
        expr[5] = 10.0
        expr[10] = 5.0
        imp = iface.gene_importance(expr)
        # Only expressed genes should have non-zero importance
        for i in range(20):
            if i not in (5, 10):
                assert imp[i] == pytest.approx(0.0)

    def test_sigma_controls_attention_sharpness(self) -> None:
        """Small sigma → sharper attention → more concentrated importance."""
        rng = np.random.default_rng(42)
        expr = rng.poisson(3, 50).astype(np.float64)
        iface_sharp = ScKGBERTInterface(d_model=8, n_genes=50, sigma=0.1, seed=1)
        iface_broad = ScKGBERTInterface(d_model=8, n_genes=50, sigma=10.0, seed=1)
        imp_sharp = iface_sharp.gene_importance(expr)
        imp_broad = iface_broad.gene_importance(expr)
        # Sharper attention → higher max importance (more concentrated)
        nonzero_sharp = imp_sharp[imp_sharp > 0]
        nonzero_broad = imp_broad[imp_broad > 0]
        if len(nonzero_sharp) > 0 and len(nonzero_broad) > 0:
            cv_sharp = nonzero_sharp.std() / (nonzero_sharp.mean() + 1e-10)
            cv_broad = nonzero_broad.std() / (nonzero_broad.mean() + 1e-10)
            assert cv_sharp > cv_broad
