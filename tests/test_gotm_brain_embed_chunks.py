# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEmbedChunks from former test_gotm_brain.py

"""Focused suite: TestEmbedChunks from former test_gotm_brain.py."""

from __future__ import annotations

from tests.gotm_brain_support import *  # noqa: F403

class TestEmbedChunks:
    def test_shape(self) -> None:
        chunks = [
            ContentChunk("R", "a.py", 0, "hello world test", "code", 1.0),
            ContentChunk("R", "b.md", 0, "documentation text", "markdown", 1.2),
        ]
        vectors = embed_chunks(chunks, n_dims=32)
        assert vectors.shape == (2, 32)

    def test_values_normalised(self) -> None:
        chunks = [ContentChunk("R", "a.py", 0, "test " * 100, "code", 1.0)]
        vectors = embed_chunks(chunks, n_dims=32)
        assert np.all(vectors >= 0.0)
        assert np.all(vectors <= 1.0)

    def test_deterministic(self) -> None:
        chunks = [ContentChunk("R", "a.py", 0, "deterministic test", "code", 1.0)]
        v1 = embed_chunks(chunks, seed=42)
        v2 = embed_chunks(chunks, seed=42)
        np.testing.assert_array_equal(v1, v2)

    def test_different_content_different_vectors(self) -> None:
        c1 = [ContentChunk("R", "a.py", 0, "aaaa" * 50, "code", 1.0)]
        c2 = [ContentChunk("R", "b.py", 0, "zzzz" * 50, "code", 1.0)]
        v1 = embed_chunks(c1)
        v2 = embed_chunks(c2)
        assert not np.allclose(v1, v2)

    def test_low_dimension_embeddings_preserve_requested_shape(self) -> None:
        chunk = ContentChunk("R", "empty.md", 0, "", "metadata", 0.3)
        zero_dim = embed_chunks([chunk], n_dims=0)
        assert zero_dim.shape == (1, 0)

        one_dim = embed_chunks([chunk], n_dims=1)
        assert one_dim.shape == (1, 1)
        assert np.all(one_dim == 0.0)

    def test_feature_dimensions_encode_weight_type_and_hash(self) -> None:
        chunks = [
            ContentChunk("R", "doc.py", 0, "alpha beta gamma", "docstring", 3.0),
            ContentChunk("R", "unknown.dat", 0, "alpha beta gamma", "custom", 1.0),
        ]
        vectors = embed_chunks(chunks, n_dims=32)
        assert vectors[0, 26] > 0.0
        assert vectors[0, 27] == pytest.approx(1.0)
        assert vectors[0, 28] == 1.0
        assert vectors[0, 29] == pytest.approx(0.9)
        assert vectors[1, 29] == pytest.approx(0.5)
        assert np.all((vectors[:, 30:32] >= 0.0) & (vectors[:, 30:32] <= 1.0))
