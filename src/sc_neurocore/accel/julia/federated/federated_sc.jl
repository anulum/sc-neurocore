# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for federated/federated_sc

module FederatedScAccel

using Statistics, LinearAlgebra

mutable struct AuditLogState
    epsilon::Float64
    sensitivity::Float64
    target_epsilon::Float64
    target_delta::Float64
    alpha::Float64
    rdp_budget::Float64
    rounds_consumed::Float64
    num_parties::Float64
    bitstream_length::Float64
    dp::Float64
    client_id::Float64
    encoder::Float64
    rng::Float64
    local_weights::Float64
    commitment::Float64
end

function AuditLogState()
    AuditLogState(0.0, 1.0, 10.0, 1e-05, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function lfsr_encode(value, seed, length)
    threshold = int(clamp(value, 0.0, 1.0) * 65535)
    reg = seed & 0xFFFF
    if reg == 0
        reg = 1
    bits = zeros(length, dtype=np.uint8)
    for i in 1:length
        bits[i] = 1 if reg < threshold else 0
        reg = _lfsr16_step(reg)
    return bits
end

function bitstream_probability(bits)
    n = length(bits)
    return float(sum(bits)) / n if n > 0 else 0.0
end

function flip_probability(s::AuditLogState)
    e = s.epsilon / s.sensitivity
    return 1.0 / (1.0 + math.exp(e))
end

function privatise(s::AuditLogState, bitstream, rng)
    p = s.flip_probability
    flip_mask = rng.random(length(bitstream)) < p
    noisy = bitstream.copy()
    noisy[flip_mask] = 1 - noisy[flip_mask]
    return noisy
end

function per_bit_epsilon(s::AuditLogState)
    p = s.flip_probability
    if p <= 0 || p >= 1
        return float("inf")
    return abs(math.log((1 - p) / p))
end

function total_epsilon(s::AuditLogState, bitstream_length)
    eps_1 = s.per_bit_epsilon()
    return eps_1 * math.sqrt(2 * bitstream_length * math.log(1 / 1e-5))
end

function clip_gradients(gradients, max_norm)
    l2 = float(norm(gradients))
    if l2 > max_norm && l2 > 0
        return gradients * (max_norm / l2)
    return gradients.copy()
end

function sparsify_topk(gradients, k)
    k = min(k, length(gradients))
    indices = np.argsort(abs(gradients))[-k:]
    mask = zeros(length(gradients), dtype=np.uint8)
    mask[indices] = 1
    sparse = np.zeros_like(gradients)
    sparse[indices] = gradients[indices]
    return sparse, mask
end

function consume_round(s::AuditLogState, mechanism, bitstream_length)
    eps_1 = mechanism.per_bit_epsilon()
    # RDP of randomized response at order alpha
    rdp_step = (
        (s.alpha / (s.alpha - 1))
        * math.log(
            (1 - mechanism.flip_probability) ^ s.alpha
            + mechanism.flip_probability^s.alpha
        )
        * bitstream_length
    )
    s.rdp_budget += abs(rdp_step)
    s.rounds_consumed += 1
    return ! s.is_exhausted()
end

function current_epsilon(s::AuditLogState)
    if s.rdp_budget <= 0
        return 0.0
    return s.rdp_budget + math.log(1 / s.target_delta) / (s.alpha - 1)
end

function remaining_epsilon(s::AuditLogState)
    return max(0.0, s.target_epsilon - s.current_epsilon())
end

function is_exhausted(s::AuditLogState)
    return s.current_epsilon() >= s.target_epsilon
end

function split(s::AuditLogState, bitstream, rng)
    shares = []
    accumulated = np.zeros_like(bitstream)
    for i in 1:s.num_parties - 1
        share = rng.integers(0, 2, size=length(bitstream), dtype=np.uint8)
        shares = push!(, share)
        accumulated ^= share
    # Last share ensures XOR of all shares == original
    shares = push!(, bitstream ^ accumulated)
    return shares
end

function reconstruct(s::AuditLogState)
    result = np.zeros_like(shares[0])
    for share in shares
        result ^= share
    return result
end

function verify_reconstruction(s::AuditLogState)
    return np.array_equal(original, SecretShare.reconstruct(shares))
end

function commit(s::AuditLogState)
    payload = data.tobytes()
    if nonce is ! nothing
        payload = nonce + payload
    return hashlib.sha256(payload).hexdigest()
end

function verify(s::AuditLogState)
    return CommitmentScheme.commit(data, nonce) == commitment
end

function generate_nonce(s::AuditLogState)
    return rng.bytes(32)
end

function encode(s::AuditLogState)
    self,
    gradients: np.ndarray,
    seeds: np.ndarray,
    rng: np.random.Generator,
    ) -> List[np.ndarray]
    g_min, g_max = gradients.min(), gradients.max()
    span = g_max - g_min
    if span < 1e-12
        normalised = np.full_like(gradients, 0.5)
    else
        normalised = (gradients - g_min) / span
    bitstreams = []
    for i, val in enumerate(normalised)
        seed = int(seeds[i % length(seeds)]) & 0xFFFF
        if seed == 0
            seed = 1
        bs = lfsr_encode(val, seed, s.bitstream_length)
        bs = s.dp.privatise(bs, rng)
        bitstreams = push!(, bs)
    return bitstreams
end

function decode(s::AuditLogState, bitstreams, g_min, g_max)
    probs = collect([bitstream_probability(bs) for bs in bitstreams])
    span = g_max - g_min
    return probs * span + g_min
end

function local_train(s::AuditLogState, data, labels, lr)
    if s.local_weights is nothing
        s.local_weights = s.rng.standard_normal(data.shape[1]) * 0.01
    predictions = data @ s.local_weights
    errors = predictions - labels
    gradients = 2.0 / length(labels) * (data.T @ errors)
    s.local_weights -= lr * gradients
    return gradients
end

function encode_gradients(s::AuditLogState, gradients)
    seeds = s.rng.integers(1, 65535, size=length(gradients), dtype=np.int64)
    bitstreams = s.encoder.encode(gradients, seeds, s.rng)
    # Commit to the privatised bitstreams
    concatenated = vcat(bitstreams)
    nonce = CommitmentScheme.generate_nonce(s.rng)
    s.commitment = CommitmentScheme.commit(concatenated, nonce)
    return bitstreams, s.commitment, float(gradients.min()), float(gradients.max())
end

function aggregate_bitstreams(s::AuditLogState)
    self,
    client_bitstreams: List[List[np.ndarray]],
    weights: Optional[List[float]] = nothing,
    ) -> List[np.ndarray]
    num_dims = length(client_bitstreams[0])
    n_clients = length(client_bitstreams)
    if weights is nothing
        w = ones(n_clients) / n_clients
    else
        w = collect(weights)
        w = w / w.sum()
    aggregated = []
    for dim in 1:num_dims
        stacked = np.stack([c[dim] for c in client_bitstreams]).astype(np.float64)
        weighted_sum = w @ stacked
        agg_bs = (weighted_sum > 0.5).astype(np.uint8)
        aggregated = push!(, agg_bs)
    return aggregated
end

function detect_outliers(s::AuditLogState)
    self,
    client_bitstreams: List[List[np.ndarray]],
    threshold: float = 0.3,
    ) -> List[bool]
    n = length(client_bitstreams)
    if n < 2
        return [false] * n
    # Flatten each client's update to a single vector
    flat = []
    for cbs in client_bitstreams
        flat = push!(, vcat(cbs).astype(np.float64))
    is_outlier = []
    for i in 1:n
        sims = []
        for j in 1:n
            if i == j
                continue
            dot = dot(flat[i], flat[j])
            na = norm(flat[i])
            nb = norm(flat[j])
            if na > 0 && nb > 0
                sims = push!(, dot / (na * nb))
            else
                sims = push!(, 0.0)
        mean_sim = float(mean(sims))
        is_outlier = push!(, mean_sim < threshold)
    return is_outlier
end

function verify_commitments(s::AuditLogState)
    self,
    client_bitstreams: List[List[np.ndarray]],
    commitments: List[str],
    nonces: Optional[List[bytes]] = nothing,
    ) -> List[bool]
    results = []
    for i, (bs_list, commitment) in enumerate(zip(client_bitstreams, commitments))
        concatenated = vcat(bs_list)
        nonce = nonces[i] if nonces else nothing
        results = push!(, CommitmentScheme.commit(concatenated, nonce) == commitment)
    return results
end

function poisson_subsample(clients, sampling_rate, rng)
    clients: List[FederatedClient],
    sampling_rate: float,
    rng: np.random.Generator,
    ) -> List[FederatedClient]
    selected = []
    for c in clients
        if rng.random() < sampling_rate
            selected = push!(, c)
    return selected if selected else [clients[0]]
end

function record(s::AuditLogState, aggregated_gradient)
    s.grad_norms = push!(, float(norm(aggregated_gradient)))
end

function record_loss(s::AuditLogState, loss)
    s.round_losses = push!(, loss)
end

function converged(s::AuditLogState)
    if length(s.grad_norms) < 5
        return false
    return all(g < 0.01 for g in s.grad_norms[-5:])
end

function trend(s::AuditLogState)
    if length(s.grad_norms) < 2
        return "insufficient_data"
    if s.grad_norms[-1] < s.grad_norms[-2]
        return "decreasing"
    elseif s.grad_norms[-1] > s.grad_norms[-2]
        return "increasing"
    return "stable"
end

function run(s::AuditLogState)
    self,
    data_per_client: List[np.ndarray],
    labels_per_client: List[np.ndarray],
    client_weights: Optional[List[float]] = nothing,
    ) -> Optional[np.ndarray]
    if s.accountant.is_exhausted()
        return nothing
    s.round_number += 1
    # Client subsampling
    if s.sampling_rate < 1.0
        rng = np.random.default_rng(s.round_number)
        active = poisson_subsample(s.clients, s.sampling_rate, rng)
        active_indices = [s.clients.index(c) for c in active]
    else
        active = s.clients
        active_indices = list(range(length(s.clients)))
    all_bitstreams = []
    all_commitments = []
    g_mins, g_maxs = [], []
    for idx, client in zip(active_indices, active)
        gradients = client.local_train(data_per_client[idx], labels_per_client[idx])
        if s.clip_norm > 0
            gradients = clip_gradients(gradients, s.clip_norm)
        bitstreams, commitment, g_min, g_max = client.encode_gradients(gradients)
        all_bitstreams = push!(, bitstreams)
        all_commitments = push!(, commitment)
        g_mins = push!(, g_min)
        g_maxs = push!(, g_max)
    # Track privacy budget
    dp_mech = active[0].encoder.dp
    bl = active[0].encoder.bitstream_length
    s.accountant.consume_round(dp_mech, bl)
    # Select weights for active clients
    if client_weights is ! nothing
        active_weights = [client_weights[i] for i in active_indices]
    else
        active_weights = nothing
    # Aggregate
    aggregated_bs = s.aggregator.aggregate_bitstreams(all_bitstreams, weights=active_weights)
    # Decode using global min/max range
    global_min = min(g_mins)
    global_max = max(g_maxs)
    aggregated_grads = active[0].encoder.decode(aggregated_bs, global_min, global_max)
    # Track convergence
    s.convergence.record(aggregated_grads)
    # Audit log
    if s.audit_log is ! nothing
        s.audit_log.log_round(
            round_number=s.round_number,
            num_active=length(active),
            epsilon_consumed=s.accountant.current_epsilon(),
            grad_norm=float(norm(aggregated_grads)),
        )
    return aggregated_grads
end

function status(s::AuditLogState)
    return {
        "round": s.round_number,
        "epsilon_consumed": s.accountant.current_epsilon(),
        "epsilon_remaining": s.accountant.remaining_epsilon(),
        "rounds_consumed": s.accountant.rounds_consumed,
        "budget_exhausted": s.accountant.is_exhausted(),
        "converged": s.convergence.converged,
        "trend": s.convergence.trend,
    }
end

function from_accountant(s::AuditLogState)
    cls,
    accountant: PrivacyAccountant,
    mechanism: DPMechanism,
    bitstream_length: int,
    ) -> DPCertificate
    return cls(
        mechanism="bitstream_flip_rr",
        epsilon=accountant.current_epsilon(),
        delta=accountant.target_delta,
        rounds=accountant.rounds_consumed,
        bitstream_length=bitstream_length,
        composition_method="renyi_dp",
        accountant_state={
            "rdp_budget": accountant.rdp_budget,
            "alpha": accountant.alpha,
            "target_epsilon": accountant.target_epsilon,
            "flip_probability": mechanism.flip_probability,
        },
    )
end

function to_dict(s::AuditLogState)
    return {
        "mechanism": s.mechanism,
        "epsilon": s.epsilon,
        "delta": s.delta,
        "rounds": s.rounds,
        "bitstream_length": s.bitstream_length,
        "composition_method": s.composition_method,
        "accountant_state": s.accountant_state,
        "compliant": s.epsilon <= s.accountant_state.get("target_epsilon", float("inf")),
    }
end

function is_compliant(s::AuditLogState)
    return s.epsilon <= s.accountant_state.get("target_epsilon", float("inf"))
end

function stochastic_quantize(gradients, levels, rng)
    gradients: np.ndarray,
    levels: int,
    rng: np.random.Generator,
    ) -> np.ndarray
    g_min, g_max = gradients.min(), gradients.max()
    span = g_max - g_min
    if span < 1e-12
        return gradients.copy()
    normalised = (gradients - g_min) / span * (levels - 1)
    lower = np.floor(normalised).astype(np.int32)
    prob = normalised - lower
    upper = lower + (rng.random(length(gradients)) < prob).astype(np.int32)
    upper = clamp(upper, 0, levels - 1)
    return upper.astype(np.float64) / (levels - 1) * span + g_min
end

function step(s::AuditLogState, converging)
    if converging
        s.current_epsilon = max(
            s.min_epsilon,
            s.current_epsilon * s.decay_rate,
        )
    else
        s.current_epsilon = min(
            s.base_epsilon,
            s.current_epsilon / s.decay_rate,
        )
    return s.current_epsilon
end

function krum_select(client_vectors, num_byzantine)
    client_vectors: List[np.ndarray],
    num_byzantine: int = 1,
    ) -> int
    n = length(client_vectors)
    k = n - num_byzantine - 2
    if k < 1
        k = 1
    scores = []
    for i in 1:n
        dists = []
        for j in 1:n
            if i == j
                continue
            dists = push!(, float(norm(client_vectors[i] - client_vectors[j]) ^ 2))
        dists.sort()
        scores = push!(, sum(dists[:k]))
    return int(argmin(scores))
end

function trimmed_mean(client_vectors, trim_fraction)
    client_vectors: List[np.ndarray],
    trim_fraction: float = 0.1,
    ) -> np.ndarray
    stacked = np.stack(client_vectors)
    n = stacked.shape[0]
    trim_count = max(1, int(n * trim_fraction))
    sorted_vals = sort(stacked, axis=0)
    trimmed = sorted_vals[trim_count : n - trim_count, :]
    if trimmed.shape[0] == 0
        return mean(stacked, axis=0)
    return mean(trimmed, axis=0)
end

function fedprox_gradient(gradients, local_weights, global_weights, mu)
    gradients: np.ndarray,
    local_weights: np.ndarray,
    global_weights: np.ndarray,
    mu: float = 0.01,
    ) -> np.ndarray
    return gradients + mu * (local_weights - global_weights)
end

function accumulate(s::AuditLogState, gradients)
    if s.residual is ! nothing
        return gradients + s.residual
    return gradients.copy()
end

function update(s::AuditLogState, original, sparse)
    s.residual = original - sparse
end

function amplified_epsilon(base_epsilon, sampling_rate)
    base_epsilon: float,
    sampling_rate: float,
    ) -> float
    if sampling_rate >= 1.0
        return base_epsilon
    if sampling_rate <= 0.0
        return 0.0
    return math.log(1 + sampling_rate * (math.exp(base_epsilon) - 1))
end

function log_round(s::AuditLogState)
    self,
    round_number: int,
    num_active: int,
    epsilon_consumed: float,
    grad_norm: float,
    ) -> nothing
    s.entries = push!(,
        AuditEntry(
            round_number=round_number,
            num_active_clients=num_active,
            epsilon_consumed=epsilon_consumed,
            grad_norm=grad_norm,
        )
    )
end

function to_list(s::AuditLogState)
    return [
        {
            "round": e.round_number,
            "active_clients": e.num_active_clients,
            "epsilon": e.epsilon_consumed,
            "grad_norm": e.grad_norm,
            "timestamp": e.timestamp,
        }
        for e in s.entries
    ]
end

function total_rounds(s::AuditLogState)
    return length(s.entries)
end

function max_epsilon(s::AuditLogState)
    if ! s.entries
        return 0.0
    return max(e.epsilon_consumed for e in s.entries)
end

end # module FederatedScAccel
