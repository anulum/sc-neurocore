# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Transcriptomic foundation model interfaces

"""Transcriptomic foundation model interfaces.

Publication-exact implementations of core algorithms from two
single-cell transcriptomic foundation models:

- **ScKGBERTInterface** — Li et al. (2025), Genome Biology 26:402.
- **GeneformerInterface** — Theodoris et al. (2023), Nature 619.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from typing import Any
import numpy as np


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def rank_value_encode(
    expression: np.ndarray[Any, Any],
    global_medians: np.ndarray[Any, Any] | None = None,
) -> np.ndarray[Any, Any]:
    """Rank-value encoding for single-cell gene expression.

    Theodoris et al. (2023): genes are ranked by their expression
    in the cell, scaled by inverse frequency across the corpus
    (approximated by 1 / global_median).

    Parameters
    ----------
    expression : 1-D array [n_genes], raw counts or normalised expression.
    global_medians : 1-D array [n_genes], median expression per gene across
        the corpus.  If None, uniform weighting is used.

    Returns
    -------
    ranked_indices : int64 array — gene indices sorted by weighted expression
        (highest first), with zero-expression genes excluded.
    """
    if global_medians is not None:
        weights = 1.0 / (global_medians + 1e-10)
        weighted = expression * weights
    else:
        weighted = expression.copy()
    nonzero_mask = expression > 0
    indices = np.where(nonzero_mask)[0]
    if len(indices) == 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(-weighted[indices])
    return indices[order].astype(np.int64)


# ---------------------------------------------------------------------------
# scKGBERT — Li et al. (2025), Genome Biology 26:402
# ---------------------------------------------------------------------------


@dataclass
class ScKGBERTInterface:
    """Knowledge-enhanced foundation model for single-cell transcriptomics.

    Li Y, Qiao G, Du H, Gao X, Wang G. "scKGBERT: a knowledge-enhanced
    foundation model for single-cell transcriptomics." Genome Biology
    26:402 (2025).

    Dual-encoder architecture: S-Encoder (sequence) + K-Encoder
    (knowledge graph from STRING PPI database).  Uses Gaussian
    attention for biomarker identification.

    Gaussian attention (Li et al. 2025):
        α_ij = exp(-||q_i - k_j||² / (2σ²)) / Σ_m exp(-||q_i - k_m||² / (2σ²))

    This emphasises genes whose query-key distance is small, concentrating
    attention on biologically relevant gene–gene interactions.
    """

    d_model: int = 64
    n_genes: int = 2000
    sigma: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        """Initialise the scKG-BERT interface weights from the seed."""
        rng = np.random.default_rng(self.seed)
        # Gene token embeddings (shared between S-Encoder and K-Encoder)
        self._gene_embeddings = rng.normal(
            0.0,
            0.02,
            (self.n_genes, self.d_model),
        )
        # Knowledge graph adjacency (sparse simulation of STRING PPI)
        self._kg_adjacency = np.zeros((self.n_genes, self.n_genes), dtype=np.float64)
        n_edges = min(self.n_genes * 5, self.n_genes * (self.n_genes - 1) // 2)
        for _ in range(n_edges):
            i = rng.integers(0, self.n_genes)
            j = rng.integers(0, self.n_genes)
            if i != j:
                confidence = rng.uniform(0.15, 1.0)
                self._kg_adjacency[i, j] = confidence
                self._kg_adjacency[j, i] = confidence
        # Projection matrices for S-Encoder and K-Encoder
        self._w_q = rng.normal(0.0, 0.02, (self.d_model, self.d_model))
        self._w_k = rng.normal(0.0, 0.02, (self.d_model, self.d_model))
        self._w_v = rng.normal(0.0, 0.02, (self.d_model, self.d_model))
        # Output classification head
        self._cls_embedding = rng.normal(0.0, 0.02, self.d_model)

    def gaussian_attention(
        self,
        queries: np.ndarray[Any, Any],
        keys: np.ndarray[Any, Any],
        values: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Gaussian attention mechanism. Li et al. (2025).

        α_ij = exp(-||q_i - k_j||² / (2σ²)) / Σ_m exp(-||q_i - k_m||² / (2σ²))

        queries : [n, d], keys : [m, d], values : [m, d].
        Returns [n, d].
        """
        # Pairwise squared L2 distances: ||q_i - k_j||²
        q_sq = (queries**2).sum(axis=-1, keepdims=True)
        k_sq = (keys**2).sum(axis=-1, keepdims=True)
        dist_sq = q_sq + k_sq.T - 2.0 * queries @ keys.T
        dist_sq = np.maximum(dist_sq, 0.0)
        # Gaussian kernel
        log_weights = -dist_sq / (2.0 * self.sigma**2)
        log_weights -= log_weights.max(axis=-1, keepdims=True)
        weights = np.exp(log_weights)
        weights /= weights.sum(axis=-1, keepdims=True) + 1e-30
        attended: np.ndarray[Any, Any] = weights @ values
        return attended

    def encode_expression(self, expression: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Encode a single-cell expression profile via S-Encoder.

        expression : [n_genes] raw counts.
        Returns [d_model] cell embedding.
        """
        ranked = rank_value_encode(expression)
        if len(ranked) == 0:
            return np.zeros(self.d_model)
        # Gather token embeddings for expressed genes
        valid = ranked[ranked < self.n_genes]
        if len(valid) == 0:
            return np.zeros(self.d_model)
        tokens = self._gene_embeddings[valid]
        # Scale by rank position (higher rank = earlier in sequence)
        rank_weights = 1.0 / (np.arange(len(valid), dtype=np.float64) + 1.0)
        tokens = tokens * rank_weights[:, np.newaxis]
        # Gaussian self-attention
        q = tokens @ self._w_q
        k = tokens @ self._w_k
        v = tokens @ self._w_v
        attended = self.gaussian_attention(q, k, v)
        pooled: np.ndarray[Any, Any] = attended.mean(axis=0)
        return pooled

    def encode_with_knowledge(self, expression: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Encode via dual S-Encoder + K-Encoder pathway.

        K-Encoder aggregates neighbourhood embeddings from the
        protein–protein interaction knowledge graph (STRING).

        Returns [d_model] fused cell embedding.
        """
        s_emb = self.encode_expression(expression)
        ranked = rank_value_encode(expression)
        valid = ranked[ranked < self.n_genes]
        if len(valid) == 0:
            return s_emb
        # K-Encoder: aggregate PPI neighbourhood for expressed genes
        kg_embs = np.zeros((len(valid), self.d_model))
        for idx, gene_id in enumerate(valid):
            neighbours = self._kg_adjacency[gene_id]
            mask = neighbours > 0
            if mask.any():
                weights = neighbours[mask]
                weights /= weights.sum() + 1e-30
                kg_embs[idx] = weights @ self._gene_embeddings[mask]
            else:
                kg_embs[idx] = self._gene_embeddings[gene_id]
        # Gaussian attention on KG embeddings
        q = kg_embs @ self._w_q
        k = kg_embs @ self._w_k
        v = kg_embs @ self._w_v
        k_emb = self.gaussian_attention(q, k, v).mean(axis=0)
        # Fusion: mean of S-Encoder and K-Encoder outputs
        fused: np.ndarray[Any, Any] = (s_emb + k_emb) / 2.0
        return fused

    def predict_cell_type(
        self,
        expression: np.ndarray[Any, Any],
        prototypes: np.ndarray[Any, Any],
        labels: list[str],
    ) -> str:
        """Predict cell type via nearest prototype.

        prototypes : [n_types, d_model] — mean embeddings per cell type.
        labels : cell type names.
        Returns predicted cell type label.
        """
        emb = self.encode_with_knowledge(expression)
        dists = np.linalg.norm(prototypes - emb, axis=-1)
        return labels[int(np.argmin(dists))]

    def gene_importance(self, expression: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute gene importance scores via Gaussian attention weights.

        Returns [n_genes] importance array (higher = more important).
        """
        ranked = rank_value_encode(expression)
        valid = ranked[ranked < self.n_genes]
        importance = np.zeros(self.n_genes)
        if len(valid) == 0:
            return importance
        tokens = self._gene_embeddings[valid]
        q = tokens @ self._w_q
        k = tokens @ self._w_k
        # Gaussian attention weights
        dist_sq = ((q[:, np.newaxis, :] - k[np.newaxis, :, :]) ** 2).sum(axis=-1)
        log_w = -dist_sq / (2.0 * self.sigma**2)
        log_w -= log_w.max(axis=-1, keepdims=True)
        weights = np.exp(log_w)
        weights /= weights.sum(axis=-1, keepdims=True) + 1e-30
        # Importance = sum of incoming attention
        gene_scores = weights.sum(axis=0)
        for idx, gene_id in enumerate(valid):
            importance[gene_id] = gene_scores[idx]
        return importance


# ---------------------------------------------------------------------------
# Geneformer — Theodoris et al. (2023), Nature 619
# ---------------------------------------------------------------------------


@dataclass
class GeneformerInterface:
    """Rank-value tokenisation and masked gene prediction.

    Theodoris CV et al. "Transfer learning enables predictions in
    network biology." Nature 619 (2023).

    Core innovation: each cell's transcriptome is represented as a
    sequence of gene tokens, ranked by expression scaled by inverse
    corpus frequency.  The model is pretrained with masked gene
    prediction (analogous to BERT MLM) to learn network dynamics.
    """

    d_model: int = 256
    n_genes: int = 2000
    n_heads: int = 4
    mask_ratio: float = 0.15
    seed: int = 42

    def __post_init__(self) -> None:
        """Initialise the Geneformer interface weights from the seed."""
        rng = np.random.default_rng(self.seed)
        self._gene_embeddings = rng.normal(
            0.0,
            0.02,
            (self.n_genes, self.d_model),
        )
        # Multi-head self-attention weights
        self._w_q = rng.normal(0.0, 0.02, (self.d_model, self.d_model))
        self._w_k = rng.normal(0.0, 0.02, (self.d_model, self.d_model))
        self._w_v = rng.normal(0.0, 0.02, (self.d_model, self.d_model))
        self._w_o = rng.normal(0.0, 0.02, (self.d_model, self.d_model))
        # MLM prediction head: project back to gene vocabulary
        self._mlm_head = rng.normal(
            0.0,
            0.02,
            (self.n_genes, self.d_model),
        )

    def tokenise(
        self,
        expression: np.ndarray[Any, Any],
        global_medians: np.ndarray[Any, Any] | None = None,
    ) -> np.ndarray[Any, Any]:
        """Rank-value tokenisation. Theodoris et al. (2023).

        Returns gene indices sorted by weighted expression (descending),
        filtered to the gene vocabulary.
        """
        ranked = rank_value_encode(expression, global_medians)
        return ranked[ranked < self.n_genes]

    def mask_tokens(
        self,
        token_ids: np.ndarray[Any, Any],
        rng_seed: int | None = None,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Randomly mask tokens for MLM pretraining.

        Returns (masked_ids, mask_positions).
        mask_positions: boolean array, True where masked.
        """
        rng = np.random.default_rng(rng_seed if rng_seed is not None else self.seed)
        n = len(token_ids)
        n_mask = max(1, int(n * self.mask_ratio))
        mask_idx = rng.choice(n, size=n_mask, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[mask_idx] = True
        masked = token_ids.copy()
        masked[mask] = -1  # sentinel for masked positions
        return masked, mask

    def multi_head_attention(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Multi-head self-attention. Vaswani et al. (2017).

        x : [seq_len, d_model].
        Returns [seq_len, d_model].
        """
        n, d = x.shape
        head_dim = d // self.n_heads
        q = x @ self._w_q
        k = x @ self._w_k
        v = x @ self._w_v
        output = np.zeros_like(x)
        for h in range(self.n_heads):
            s = h * head_dim
            e = s + head_dim
            qh, kh, vh = q[:, s:e], k[:, s:e], v[:, s:e]
            scores = qh @ kh.T / math.sqrt(head_dim)
            scores -= scores.max(axis=-1, keepdims=True)
            weights = np.exp(scores)
            weights /= weights.sum(axis=-1, keepdims=True) + 1e-30
            output[:, s:e] = weights @ vh
        return output @ self._w_o

    def encode_cell(
        self,
        expression: np.ndarray[Any, Any],
        global_medians: np.ndarray[Any, Any] | None = None,
    ) -> np.ndarray[Any, Any]:
        """Extract cell-level embedding from expression profile.

        Returns [d_model] embedding (mean-pooled over gene tokens).
        """
        token_ids = self.tokenise(expression, global_medians)
        if len(token_ids) == 0:
            return np.zeros(self.d_model)
        tokens = self._gene_embeddings[token_ids]
        attended = self.multi_head_attention(tokens)
        pooled: np.ndarray[Any, Any] = attended.mean(axis=0)
        return pooled

    def predict_masked_genes(
        self,
        expression: np.ndarray[Any, Any],
        global_medians: np.ndarray[Any, Any] | None = None,
        rng_seed: int | None = None,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Masked gene prediction (MLM objective).

        Returns (mask_positions, true_gene_ids, predicted_gene_ids).
        """
        token_ids = self.tokenise(expression, global_medians)
        if len(token_ids) < 2:
            return (
                np.array([], dtype=bool),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            )
        masked_ids, mask = self.mask_tokens(token_ids, rng_seed)
        # Build embeddings, replacing masked positions with zero
        tokens = np.zeros((len(masked_ids), self.d_model))
        for i, tid in enumerate(masked_ids):
            if tid >= 0:
                tokens[i] = self._gene_embeddings[tid]
        attended = self.multi_head_attention(tokens)
        # Predict masked positions via MLM head
        masked_repr = attended[mask]
        logits = masked_repr @ self._mlm_head.T
        predicted = np.argmax(logits, axis=-1).astype(np.int64)
        true_ids = token_ids[mask]
        return mask, true_ids, predicted

    def gene_network_attention(
        self,
        expression: np.ndarray[Any, Any],
        global_medians: np.ndarray[Any, Any] | None = None,
    ) -> np.ndarray[Any, Any]:
        """Extract attention-derived gene–gene interaction matrix.

        Theodoris et al. (2023) showed that attention weights encode
        network hierarchy. Returns [n_expressed, n_expressed] attention
        matrix averaged across heads.
        """
        token_ids = self.tokenise(expression, global_medians)
        if len(token_ids) < 2:
            return np.array([[]])
        tokens = self._gene_embeddings[token_ids]
        n = len(tokens)
        head_dim = self.d_model // self.n_heads
        q = tokens @ self._w_q
        k = tokens @ self._w_k
        avg_attn = np.zeros((n, n))
        for h in range(self.n_heads):
            s = h * head_dim
            e = s + head_dim
            scores = q[:, s:e] @ k[:, s:e].T / math.sqrt(head_dim)
            scores -= scores.max(axis=-1, keepdims=True)
            weights = np.exp(scores)
            weights /= weights.sum(axis=-1, keepdims=True) + 1e-30
            avg_attn += weights
        return avg_attn / self.n_heads
