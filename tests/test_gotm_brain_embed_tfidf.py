# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEmbedTfidf from former test_gotm_brain.py

"""Focused suite: TestEmbedTfidf from former test_gotm_brain.py."""

from __future__ import annotations

from tests.gotm_brain_support import *  # noqa: F403


class TestEmbedTfidf:
    def test_empty_corpus_returns_empty_matrix_and_vocab(self) -> None:
        matrix, vocab = embed_tfidf([], n_dims=7)
        assert matrix.shape == (0, 7)
        assert vocab == {}

    def test_terms_filtered_out_returns_zero_matrix(self) -> None:
        chunks = [
            ContentChunk("R", "a.md", 0, "single unique alpha", "markdown", 1.0),
            ContentChunk("R", "b.md", 0, "another unique beta", "markdown", 1.0),
        ]
        matrix, vocab = embed_tfidf(chunks, n_dims=5, min_df=3)
        assert matrix.shape == (2, 5)
        assert vocab == {}
        assert np.all(matrix == 0.0)

    def test_corpus_tfidf_stems_stopwords_filters_and_l2_normalises(self) -> None:
        chunks = [
            ContentChunk(
                "R",
                "quantum_a.md",
                0,
                "the the fisher posner binding oscillation return",
                "markdown",
                1.0,
            ),
            ContentChunk(
                "R",
                "quantum_b.md",
                0,
                "fisher posner binding oscillations coherence",
                "markdown",
                1.0,
            ),
            ContentChunk(
                "R",
                "metabolic.md",
                0,
                "fisher atp metabolism coherence",
                "markdown",
                1.0,
            ),
        ]
        matrix, vocab = embed_tfidf(chunks, n_dims=8, min_df=2, max_df_ratio=0.85)

        assert matrix.shape == (3, 8)
        assert "the" not in vocab
        assert "return" not in vocab
        assert "fisher" not in vocab  # appears in every document, above max_df_ratio
        assert {"posn", "bind", "coherence"} <= set(vocab)

        row_norms = np.linalg.norm(matrix[:, : len(vocab)], axis=1)
        assert np.all(row_norms > 0.0)
        np.testing.assert_allclose(row_norms, np.ones_like(row_norms))
