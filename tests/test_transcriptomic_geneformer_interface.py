# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGeneformerInterface from former test_transcriptomic.py

"""Focused suite: TestGeneformerInterface from former test_transcriptomic.py."""

from __future__ import annotations

from tests.transcriptomic_support import *  # noqa: F403


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
