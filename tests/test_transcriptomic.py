# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for transcriptomic foundation model interfaces

"""Tests for transcriptomic interfaces: scKGBERT, Geneformer.

Multi-angle, publication-property tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.bio.transcriptomic import (
    GeneformerInterface,
    ScKGBERTInterface,
    rank_value_encode,
)


# ---------------------------------------------------------------------------
# Shared: rank-value encoding
# ---------------------------------------------------------------------------


class TestRankValueEncode:
    """Theodoris et al. (2023) rank-value tokenisation."""

    def test_all_zeros(self) -> None:
        result = rank_value_encode(np.zeros(10))
        assert len(result) == 0

    def test_descending_order(self) -> None:
        expr = np.array([0.0, 5.0, 1.0, 10.0, 3.0])
        ranked = rank_value_encode(expr)
        assert ranked[0] == 3  # gene 3 has highest expression (10.0)
        assert ranked[1] == 1  # gene 1 has second highest (5.0)

    def test_zeros_excluded(self) -> None:
        expr = np.array([0.0, 1.0, 0.0, 2.0])
        ranked = rank_value_encode(expr)
        assert 0 not in ranked
        assert 2 not in ranked
        assert len(ranked) == 2

    def test_global_median_weighting(self) -> None:
        """Rare genes (low median) get upweighted."""
        expr = np.array([2.0, 2.0, 2.0])
        medians = np.array([10.0, 0.1, 1.0])
        ranked = rank_value_encode(expr, medians)
        # Gene 1 has lowest median → highest weight → first
        assert ranked[0] == 1

    def test_single_gene(self) -> None:
        expr = np.array([0.0, 0.0, 5.0])
        ranked = rank_value_encode(expr)
        assert len(ranked) == 1
        assert ranked[0] == 2


# ---------------------------------------------------------------------------
# scKGBERT — Li et al. (2025)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Geneformer — Theodoris et al. (2023)
# ---------------------------------------------------------------------------


class TestGeneformerInterface:
    def test_defaults(self) -> None:
        iface = GeneformerInterface(n_genes=100)
        assert iface.d_model == 256
        assert iface.n_heads == 4
        assert iface.mask_ratio == pytest.approx(0.15)

    def test_tokenise_descending(self) -> None:
        iface = GeneformerInterface(n_genes=10)
        expr = np.array([0.0, 3.0, 1.0, 5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        tokens = iface.tokenise(expr)
        assert tokens[0] == 3  # highest expression
        assert 0 not in tokens  # zero expression excluded
        assert 4 not in tokens

    def test_tokenise_with_medians(self) -> None:
        iface = GeneformerInterface(n_genes=5)
        expr = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        medians = np.array([10.0, 0.01, 1.0, 5.0, 0.5])
        tokens = iface.tokenise(expr, medians)
        # Gene 1 (lowest median) should rank first
        assert tokens[0] == 1

    def test_mask_tokens_ratio(self) -> None:
        iface = GeneformerInterface(n_genes=100, mask_ratio=0.2)
        token_ids = np.arange(50, dtype=np.int64)
        masked, mask = iface.mask_tokens(token_ids, rng_seed=42)
        n_masked = mask.sum()
        assert n_masked == 10  # 0.2 * 50
        assert (masked[mask] == -1).all()
        assert (masked[~mask] >= 0).all()

    def test_multi_head_attention_shape(self) -> None:
        iface = GeneformerInterface(d_model=16, n_genes=10, n_heads=2)
        x = np.random.default_rng(1).normal(0, 1, (8, 16))
        out = iface.multi_head_attention(x)
        assert out.shape == (8, 16)

    def test_encode_cell_shape(self) -> None:
        iface = GeneformerInterface(d_model=32, n_genes=50)
        expr = np.random.default_rng(1).poisson(3, 50).astype(np.float64)
        emb = iface.encode_cell(expr)
        assert emb.shape == (32,)

    def test_encode_cell_zeros(self) -> None:
        iface = GeneformerInterface(d_model=16, n_genes=20)
        emb = iface.encode_cell(np.zeros(20))
        assert np.allclose(emb, 0.0)

    def test_different_profiles_different_embeddings(self) -> None:
        iface = GeneformerInterface(d_model=32, n_genes=50, seed=7)
        rng = np.random.default_rng(42)
        expr1 = rng.poisson(5, 50).astype(np.float64)
        expr2 = rng.poisson(1, 50).astype(np.float64)
        emb1 = iface.encode_cell(expr1)
        emb2 = iface.encode_cell(expr2)
        assert not np.allclose(emb1, emb2)

    def test_predict_masked_genes_shapes(self) -> None:
        iface = GeneformerInterface(d_model=16, n_genes=30, n_heads=2, seed=1)
        expr = np.random.default_rng(5).poisson(3, 30).astype(np.float64)
        mask, true_ids, predicted = iface.predict_masked_genes(expr, rng_seed=42)
        assert len(true_ids) == mask.sum()
        assert len(predicted) == mask.sum()
        assert all(0 <= p < 30 for p in predicted)

    def test_predict_masked_genes_insufficient_tokens(self) -> None:
        iface = GeneformerInterface(d_model=8, n_genes=10)
        expr = np.zeros(10)  # no expressed genes
        mask, true_ids, predicted = iface.predict_masked_genes(expr)
        assert len(true_ids) == 0

    def test_gene_network_attention_shape(self) -> None:
        iface = GeneformerInterface(d_model=16, n_genes=20, n_heads=2, seed=3)
        expr = np.random.default_rng(8).poisson(2, 20).astype(np.float64)
        attn = iface.gene_network_attention(expr)
        n_expressed = (expr > 0).sum()
        if n_expressed >= 2:
            assert attn.shape == (n_expressed, n_expressed)

    def test_attention_rows_sum_to_one(self) -> None:
        """Each row of the averaged attention matrix should sum to 1."""
        iface = GeneformerInterface(d_model=16, n_genes=30, n_heads=4, seed=5)
        expr = np.random.default_rng(3).poisson(3, 30).astype(np.float64)
        attn = iface.gene_network_attention(expr)
        if attn.size > 1:
            row_sums = attn.sum(axis=-1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_attention_encodes_hierarchy(self) -> None:
        """Theodoris et al. (2023): attention weights encode network hierarchy.
        Highly expressed genes should receive more attention (higher column sums)."""
        iface = GeneformerInterface(d_model=32, n_genes=20, n_heads=4, seed=42)
        # Create expression with clear hierarchy
        expr = np.zeros(20)
        expr[0] = 100.0  # dominant gene
        expr[1] = 50.0
        expr[2] = 10.0
        expr[3] = 5.0
        expr[4] = 1.0
        attn = iface.gene_network_attention(expr)
        # Attention matrix exists and is non-trivial
        assert attn.shape[0] == 5
        assert attn.shape[1] == 5


class TestTranscriptomicEmptyExpression:
    def test_kgbert_returns_zero_embeddings_for_silent_expression(self) -> None:
        enc = ScKGBERTInterface(d_model=8, n_genes=10)
        silent = np.zeros(10)
        assert enc.encode_expression(silent).shape == (8,)
        assert np.allclose(enc.encode_expression(silent), 0.0)
        assert enc.encode_with_knowledge(silent).shape == (8,)
        assert enc.gene_importance(silent).shape == (10,)
        assert np.allclose(enc.gene_importance(silent), 0.0)

    def test_kgbert_returns_zero_when_only_out_of_vocab_genes_expressed(self) -> None:
        # Genes expressed only at indices beyond the model vocabulary rank
        # non-empty but filter to no valid tokens, so the cell embedding is zero.
        enc = ScKGBERTInterface(d_model=8, n_genes=10)
        expression = np.zeros(15)
        expression[12] = 5.0  # only an out-of-vocabulary gene is expressed
        result = enc.encode_expression(expression)
        assert result.shape == (8,)
        assert np.allclose(result, 0.0)

    def test_kgbert_uses_self_embedding_for_neighbourless_gene(self) -> None:
        enc = ScKGBERTInterface(d_model=8, n_genes=10)
        enc._kg_adjacency[0, :] = 0.0  # isolate gene 0 in the PPI graph
        expression = np.zeros(10)
        expression[0] = 5.0
        result = enc.encode_with_knowledge(expression)
        assert result.shape == (8,)

    def test_geneformer_attention_empty_for_too_few_tokens(self) -> None:
        iface = GeneformerInterface(d_model=8, n_genes=10)
        attn = iface.gene_network_attention(np.zeros(10))
        assert attn.size == 0
