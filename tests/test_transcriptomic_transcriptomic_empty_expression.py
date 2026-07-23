# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTranscriptomicEmptyExpression from former test_transcriptomic.py

"""Focused suite: TestTranscriptomicEmptyExpression from former test_transcriptomic.py."""

from __future__ import annotations

from tests.transcriptomic_support import *  # noqa: F403

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
