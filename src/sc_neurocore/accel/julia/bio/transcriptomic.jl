# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for bio/transcriptomic

module TranscriptomicAccel

using Statistics, LinearAlgebra

mutable struct GeneformerInterfaceState
    d_model::Float64
    n_genes::Float64
    sigma::Float64
    seed::Float64
    n_heads::Float64
    mask_ratio::Float64
end

function GeneformerInterfaceState()
    GeneformerInterfaceState(256.0, 2000.0, 1.0, 42.0, 4.0, 0.15)
end

function rank_value_encode(expression, global_medians)
    expression: np.ndarray,
    global_medians: np.ndarray | nothing = nothing,
    ) -> np.ndarray
    if global_medians is ! nothing
        weights = 1.0 / (global_medians + 1e-10)
        weighted = expression * weights
    else
        weighted = expression.copy()
    nonzero_mask = expression > 0
    indices = findall(nonzero_mask)[0]
    if length(indices) == 0
        return collect([], dtype=np.int64)
    order = np.argsort(-weighted[indices])
    return indices[order].astype(np.int64)
end

function gaussian_attention(s::GeneformerInterfaceState)
    self,
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    ) -> np.ndarray
    # Pairwise squared L2 distances: ||q_i - k_j||²
    q_sq = (queries^2).sum(axis=-1, keepdims=true)
    k_sq = (keys^2).sum(axis=-1, keepdims=true)
    dist_sq = q_sq + k_sq.T - 2.0 * queries @ keys.T
    dist_sq = max(dist_sq, 0.0)
    # Gaussian kernel
    log_weights = -dist_sq / (2.0 * s.sigma^2)
    log_weights -= log_weights.max(axis=-1, keepdims=true)
    weights = exp(log_weights)
    weights /= weights.sum(axis=-1, keepdims=true) + 1e-30
    return weights @ values
end

function encode_expression(s::GeneformerInterfaceState, expression)
    ranked = rank_value_encode(expression)
    if length(ranked) == 0
        return zeros(s.d_model)
    # Gather token embeddings for expressed genes
    valid = ranked[ranked < s.n_genes]
    if length(valid) == 0
        return zeros(s.d_model)
    tokens = s._gene_embeddings[valid]
    # Scale by rank position (higher rank = earlier in sequence)
    rank_weights = 1.0 / (collect(length(valid), dtype=np.float64) + 1.0)
    tokens = tokens * rank_weights[:, np.newaxis]
    # Gaussian self-attention
    q = tokens @ s._w_q
    k = tokens @ s._w_k
    v = tokens @ s._w_v
    attended = s.gaussian_attention(q, k, v)
    return attended.mean(axis=0)
end

function encode_with_knowledge(s::GeneformerInterfaceState, expression)
    s_emb = s.encode_expression(expression)
    ranked = rank_value_encode(expression)
    valid = ranked[ranked < s.n_genes]
    if length(valid) == 0
        return s_emb
    # K-Encoder: aggregate PPI neighbourhood for expressed genes
    kg_embs = zeros((length(valid), s.d_model))
    for idx, gene_id in enumerate(valid)
        neighbours = s._kg_adjacency[gene_id]
        mask = neighbours > 0
        if mask.any()
            weights = neighbours[mask]
            weights /= weights.sum() + 1e-30
            kg_embs[idx] = weights @ s._gene_embeddings[mask]
        else
            kg_embs[idx] = s._gene_embeddings[gene_id]
    # Gaussian attention on KG embeddings
    q = kg_embs @ s._w_q
    k = kg_embs @ s._w_k
    v = kg_embs @ s._w_v
    k_emb = s.gaussian_attention(q, k, v).mean(axis=0)
    # Fusion: mean of S-Encoder && K-Encoder outputs
    return (s_emb + k_emb) / 2.0
end

function predict_cell_type(s::GeneformerInterfaceState)
    self,
    expression: np.ndarray,
    prototypes: np.ndarray,
    labels: list[str],
    ) -> str
    emb = s.encode_with_knowledge(expression)
    dists = norm(prototypes - emb, axis=-1)
    return labels[int(argmin(dists))]
end

function gene_importance(s::GeneformerInterfaceState, expression)
    ranked = rank_value_encode(expression)
    valid = ranked[ranked < s.n_genes]
    importance = zeros(s.n_genes)
    if length(valid) == 0
        return importance
    tokens = s._gene_embeddings[valid]
    q = tokens @ s._w_q
    k = tokens @ s._w_k
    # Gaussian attention weights
    dist_sq = ((q[:, np.newaxis, :] - k[np.newaxis, :, :]) ^ 2).sum(axis=-1)
    log_w = -dist_sq / (2.0 * s.sigma^2)
    log_w -= log_w.max(axis=-1, keepdims=true)
    weights = exp(log_w)
    weights /= weights.sum(axis=-1, keepdims=true) + 1e-30
    # Importance = sum of incoming attention
    gene_scores = weights.sum(axis=0)
    for idx, gene_id in enumerate(valid)
        importance[gene_id] = gene_scores[idx]
    return importance
end

function tokenise(s::GeneformerInterfaceState)
    self,
    expression: np.ndarray,
    global_medians: np.ndarray | nothing = nothing,
    ) -> np.ndarray
    ranked = rank_value_encode(expression, global_medians)
    return ranked[ranked < s.n_genes]
end

function mask_tokens(s::GeneformerInterfaceState)
    self,
    token_ids: np.ndarray,
    rng_seed: int | nothing = nothing,
    ) -> tuple[np.ndarray, np.ndarray]
    rng = np.random.default_rng(rng_seed if rng_seed is ! nothing else s.seed)
    n = length(token_ids)
    n_mask = max(1, int(n * s.mask_ratio))
    mask_idx = rng.choice(n, size=n_mask, replace=false)
    mask = zeros(n, dtype=bool)
    mask[mask_idx] = true
    masked = token_ids.copy()
    masked[mask] = -1  # sentinel for masked positions
    return masked, mask
end

function multi_head_attention(s::GeneformerInterfaceState, x)
    n, d = x.shape
    head_dim = d // s.n_heads
    q = x @ s._w_q
    k = x @ s._w_k
    v = x @ s._w_v
    output = np.zeros_like(x)
    for h in 1:s.n_heads
        s = h * head_dim
        e = s + head_dim
        qh, kh, vh = q[:, s:e], k[:, s:e], v[:, s:e]
        scores = qh @ kh.T / math.sqrt(head_dim)
        scores -= scores.max(axis=-1, keepdims=true)
        weights = exp(scores)
        weights /= weights.sum(axis=-1, keepdims=true) + 1e-30
        output[:, s:e] = weights @ vh
    return output @ s._w_o
end

function encode_cell(s::GeneformerInterfaceState)
    self,
    expression: np.ndarray,
    global_medians: np.ndarray | nothing = nothing,
    ) -> np.ndarray
    token_ids = s.tokenise(expression, global_medians)
    if length(token_ids) == 0
        return zeros(s.d_model)
    tokens = s._gene_embeddings[token_ids]
    attended = s.multi_head_attention(tokens)
    return attended.mean(axis=0)
end

function predict_masked_genes(s::GeneformerInterfaceState)
    self,
    expression: np.ndarray,
    global_medians: np.ndarray | nothing = nothing,
    rng_seed: int | nothing = nothing,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    token_ids = s.tokenise(expression, global_medians)
    if length(token_ids) < 2
        return (
            collect([], dtype=bool),
            collect([], dtype=np.int64),
            collect([], dtype=np.int64),
        )
    masked_ids, mask = s.mask_tokens(token_ids, rng_seed)
    # Build embeddings, replacing masked positions with zero
    tokens = zeros((length(masked_ids), s.d_model))
    for i, tid in enumerate(masked_ids)
        if tid >= 0
            tokens[i] = s._gene_embeddings[tid]
    attended = s.multi_head_attention(tokens)
    # Predict masked positions via MLM head
    masked_repr = attended[mask]
    logits = masked_repr @ s._mlm_head.T
    predicted = argmax(logits, axis=-1).astype(np.int64)
    true_ids = token_ids[mask]
    return mask, true_ids, predicted
end

function gene_network_attention(s::GeneformerInterfaceState)
    self,
    expression: np.ndarray,
    global_medians: np.ndarray | nothing = nothing,
    ) -> np.ndarray
    token_ids = s.tokenise(expression, global_medians)
    if length(token_ids) < 2
        return collect([[]])
    tokens = s._gene_embeddings[token_ids]
    n = length(tokens)
    head_dim = s.d_model // s.n_heads
    q = tokens @ s._w_q
    k = tokens @ s._w_k
    avg_attn = zeros((n, n))
    for h in 1:s.n_heads
        s = h * head_dim
        e = s + head_dim
        scores = q[:, s:e] @ k[:, s:e].T / math.sqrt(head_dim)
        scores -= scores.max(axis=-1, keepdims=true)
        weights = exp(scores)
        weights /= weights.sum(axis=-1, keepdims=true) + 1e-30
        avg_attn += weights
    return avg_attn / s.n_heads
end

end # module TranscriptomicAccel
