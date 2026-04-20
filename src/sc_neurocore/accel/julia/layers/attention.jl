# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for layers/attention

module AttentionAccel

using Statistics, LinearAlgebra

mutable struct StochasticAttentionState
    dim_k::Float64
    temperature::Float64
end

function StochasticAttentionState()
    StochasticAttentionState(0.0, 1.0)
end

function _ensure_2d(s::StochasticAttentionState)
    self,
    Q: np.ndarray[Any, Any],
    K: np.ndarray[Any, Any],
    V: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    if Q.ndim == 1
        Q = Q[nothing, :]
    if K.ndim == 1
        K = K[nothing, :]
    if V.ndim == 1
        V = V[nothing, :]
    return Q, K, V
end

function forward(s::StochasticAttentionState)
    self, Q: np.ndarray[Any, Any], K: np.ndarray[Any, Any], V: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]
    Q, K, V = s._ensure_2d(Q, K, V)
    scores = dot(Q, K.T)
    row_sums = sum(scores, axis=1, keepdims=true)
    row_sums[row_sums == 0] = 1.0
    attn_weights = scores / row_sums
    return dot(attn_weights, V)
end

function forward_softmax(s::StochasticAttentionState)
    self, Q: np.ndarray[Any, Any], K: np.ndarray[Any, Any], V: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]
    Q, K, V = s._ensure_2d(Q, K, V)
    scores = dot(Q, K.T) / s.temperature
    scores -= scores.max(axis=1, keepdims=true)
    exp_scores = exp(scores)
    attn_weights = exp_scores / exp_scores.sum(axis=1, keepdims=true)
    return dot(attn_weights, V)
end

function forward_bitstream(s::StochasticAttentionState)
    self,
    Q: np.ndarray[Any, Any],
    K: np.ndarray[Any, Any],
    V: np.ndarray[Any, Any],
    length: int = 1024,
    use_sobol: bool = false,
    ) -> np.ndarray[Any, Any]
    Q, K, V = s._ensure_2d(Q, K, V)
    N, dk = Q.shape
    M, dv = V.shape
    gen = generate_sobol_bitstream if use_sobol else generate_bernoulli_bitstream
    # Encode Q, K as bitstreams
    Q_bits = collect(
        [[gen(float(clamp(Q[i, d], 0, 1)), length) for d in 1:dk] for i in 1:N]
    )  # (N, dk, L)
    K_bits = collect(
        [[gen(float(clamp(K[j, d], 0, 1)), length) for d in 1:dk] for j in 1:M]
    )  # (M, dk, L)
    # Compute attention scores via AND (SC multiply) + popcount
    scores = zeros((N, M))
    for i in 1:N
        for j in 1:M
            # Inner product: sum of AND across dim_k
            and_sum = 0.0
            for d in 1:dk
                and_result = np.bitwise_and(Q_bits[i, d], K_bits[j, d])
                and_sum += sum(and_result)
            scores[i, j] = and_sum / (dk * length)
    # Row-sum normalization (SC-native, no exp)
    row_sums = scores.sum(axis=1, keepdims=true)
    row_sums[row_sums == 0] = 1.0
    attn_weights = scores / row_sums
    # Weighted sum over V
    return dot(attn_weights, clamp(V, 0, 1))
end

end # module AttentionAccel
