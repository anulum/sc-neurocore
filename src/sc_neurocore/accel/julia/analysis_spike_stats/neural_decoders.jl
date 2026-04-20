# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/neural_decoders

module NeuralDecodersAccel

using Statistics, LinearAlgebra

mutable struct CEBRAEncoderState
    d_model::Float64
    n_latents::Float64
    seed::Float64
    d_state::Float64
    dt::Float64
    bin_size_ms::Float64
    d_input::Float64
    d_output::Float64
    temperature::Float64
    learning_rate::Float64
end

function CEBRAEncoderState()
    CEBRAEncoderState(64.0, 32.0, 42.0, 32.0, 1.0, 20.0, 64.0, 8.0, 1.0, 0.001)
end

function tokenise_spikes(spike_trains, dt)
    spike_trains: list[np.ndarray],
    dt: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]
    uids: list[int] = []
    times: list[float] = []
    for uid, train in enumerate(spike_trains)
        indices = np.flatnonzero(train)
        for idx in indices
            uids = push!(, uid)
            times = push!(, idx * dt)
    unit_ids = collect(uids, dtype=np.int64)
    timestamps = collect(times, dtype=np.float64)
    order = np.argsort(timestamps, kind="stable")
    return unit_ids[order], timestamps[order]
end

function sinusoidal_position_encode(timestamps, d_model)
    timestamps: np.ndarray,
    d_model: int,
    ) -> np.ndarray
    n = length(timestamps)
    pe = zeros((n, d_model), dtype=np.float64)
    indices = collect(0, d_model, 2, dtype=np.float64)
    divisors = 10000.0 ^ (indices / d_model)
    for k, div in enumerate(divisors)
        pe[:, 2 * k] = sin(timestamps / div)
        if 2 * k + 1 < d_model
            pe[:, 2 * k + 1] = cos(timestamps / div)
    return pe
end

function scaled_dot_product_attention(queries, keys, values)
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    ) -> np.ndarray
    d_k = keys.shape[-1]
    scores = queries @ keys.T / sqrt(d_k)
    scores -= scores.max(axis=-1, keepdims=true)
    weights = exp(scores)
    weights /= weights.sum(axis=-1, keepdims=true) + 1e-30
    return weights @ values
end

function _unit_embedding(s::CEBRAEncoderState, unit_id)
    if unit_id ! in s._unit_embeddings
        rng = np.random.default_rng(s.seed + unit_id + 1)
        s._unit_embeddings[unit_id] = rng.normal(0.0, 0.02, s.d_model)
    return s._unit_embeddings[unit_id]
end

function encode(s::CEBRAEncoderState)
    self,
    spike_trains: list[np.ndarray],
    dt: float = 1.0,
    ) -> np.ndarray
    unit_ids, timestamps = tokenise_spikes(spike_trains, dt)
    if length(unit_ids) == 0
        return zeros((s.n_latents, s.d_model))
    pe = sinusoidal_position_encode(timestamps, s.d_model)
    token_embs = collect([s._unit_embedding(u) for u in unit_ids])
    kv = token_embs + pe
    return scaled_dot_product_attention(s._latent_queries, kv, kv)
end

function decode(s::CEBRAEncoderState)
    self,
    latents: np.ndarray,
    output_queries: np.ndarray,
    ) -> np.ndarray
    return scaled_dot_product_attention(output_queries, latents, latents)
end

function reset(s::CEBRAEncoderState)
    s._unit_embeddings.clear()
    rng = np.random.default_rng(s.seed)
    s._latent_queries = rng.normal(0.0, 0.02, (s.n_latents, s.d_model))
end

function discretise(s::CEBRAEncoderState, step_dt)
    a_bar = exp(step_dt * s._A)
    a_inv = 1.0 / (s._A + 1e-30)
    b_bar = np.diag(a_bar - 1.0) @ np.diag(a_inv) @ s._B
    return a_bar, b_bar
end

function step(s::CEBRAEncoderState, x)
    a_bar, b_bar = s.discretise(s.dt)
    s._h = a_bar * s._h + b_bar @ x
    return np.real(s._C @ s._h) + s._D @ x
end

function encode_causal(s::CEBRAEncoderState)
    self,
    spike_trains: list[np.ndarray],
    dt: float = 1.0,
    ) -> np.ndarray
    s.reset()
    if ! spike_trains
        return zeros((0, s.d_model))
    n_steps = max(length(t) for t in spike_trains)
    n_units = length(spike_trains)
    # Pad spike trains to common length
    padded = zeros((n_units, n_steps), dtype=np.float64)
    for i, train in enumerate(spike_trains)
        padded[i, : length(train)] = train
    # Project population vector to d_model via fixed random projection
    rng = np.random.default_rng(s.seed + 9999)
    proj = rng.normal(0.0, 1.0 / sqrt(n_units), (s.d_model, n_units))
    outputs = zeros((n_steps, s.d_model))
    for t_idx in 1:n_steps
        x = proj @ padded[:, t_idx]
        outputs[t_idx] = s.step(x)
    return outputs
end

function reset(s::CEBRAEncoderState)
    s._h = zeros(s.d_state, dtype=np.complex128)
end

function bin_and_embed(s::CEBRAEncoderState)
    self,
    spike_trains: list[np.ndarray],
    dt: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]
    if ! spike_trains
        return zeros((0, 0)), zeros((0, s.d_model))
    n_neurons = length(spike_trains)
    samples_per_bin = max(1, int(s.bin_size_ms / dt))
    n_steps = max(length(t) for t in spike_trains)
    n_bins = n_steps // samples_per_bin
    if n_bins == 0
        return zeros((0, n_neurons)), zeros((0, s.d_model))
    binned = zeros((n_bins, n_neurons), dtype=np.float64)
    for i, train in enumerate(spike_trains)
        for b in 1:n_bins
            start = b * samples_per_bin
            end = min(start + samples_per_bin, length(train))
            binned[b, i] = train[start:end].sum()
    # Lazy init embedding projection
    if s._embed_w is nothing || s._embed_w.shape[1] != n_neurons
        rng = np.random.default_rng(s.seed)
        s._embed_w = rng.normal(
            0.0,
            1.0 / sqrt(n_neurons),
            (s.d_model, n_neurons),
        )
        s._embed_b = zeros(s.d_model)
    assert s._embed_w is ! nothing && s._embed_b is ! nothing
    embedded = binned @ s._embed_w.T + s._embed_b
    pe = sinusoidal_position_encode(
        collect(n_bins, dtype=np.float64),
        s.d_model,
    )
    embedded += pe
    return binned, embedded
end

function predict_next(s::CEBRAEncoderState, embedded)
    n = embedded.shape[0]
    if n == 0
        return zeros((0, s.d_model))
    d_k = embedded.shape[-1]
    scores = embedded @ embedded.T / sqrt(d_k)
    # Causal mask: positions can only attend to earlier positions
    mask = np.triu(np.full((n, n), -1e9), k=1)
    scores += mask
    scores -= scores.max(axis=-1, keepdims=true)
    weights = exp(scores)
    weights /= weights.sum(axis=-1, keepdims=true) + 1e-30
    attended = weights @ embedded
    return attended @ s._output_w.T + s._output_b
end

function decode(s::CEBRAEncoderState)
    self,
    spike_trains: list[np.ndarray],
    dt: float = 1.0,
    ) -> np.ndarray
    _, embedded = s.bin_and_embed(spike_trains, dt)
    return s.predict_next(embedded)
end

function encode(s::CEBRAEncoderState, x)
    squeeze = x.ndim == 1
    if squeeze
        x = x[np.newaxis, :]
    h = x @ s._w1.T + s._b1
    h = max(h, 0.0)  # ReLU
    z = h @ s._w2.T + s._b2
    # L2 normalise to unit hypersphere
    norms = norm(z, axis=-1, keepdims=true) + 1e-30
    z = z / norms
    if squeeze
        z = z[0]
    return z
end

function cosine_similarity(s::CEBRAEncoderState)
    a_norm = a / (norm(a, axis=-1, keepdims=true) + 1e-30)
    b_norm = b / (norm(b, axis=-1, keepdims=true) + 1e-30)
    return a_norm @ b_norm.T
end

function infonce_loss(s::CEBRAEncoderState)
    self,
    anchors: np.ndarray,
    positives: np.ndarray,
    ) -> float
    z_a = s.encode(anchors)
    z_p = s.encode(positives)
    # Similarity matrix: each anchor vs all positives
    sim_matrix = s.cosine_similarity(z_a, z_p) / s.temperature
    sim_matrix -= sim_matrix.max(axis=-1, keepdims=true)
    exp_sim = exp(sim_matrix)
    # Positive similarities on the diagonal
    pos_sim = np.diag(exp_sim)
    loss = -mean(log(pos_sim / (exp_sim.sum(axis=-1) + 1e-30) + 1e-30))
    return float(loss)
end

function _forward_and_loss(s::CEBRAEncoderState)
    self,
    anchors: np.ndarray,
    positives: np.ndarray,
    ) -> tuple[float, dict[str, np.ndarray]]
    # Layer 1
    h1_pre = anchors @ s._w1.T + s._b1
    h1 = max(h1_pre, 0.0)
    z1_pre = h1 @ s._w2.T + s._b2
    n1 = norm(z1_pre, axis=-1, keepdims=true) + 1e-30
    z_a = z1_pre / n1
    h2_pre = positives @ s._w1.T + s._b1
    h2 = max(h2_pre, 0.0)
    z2_pre = h2 @ s._w2.T + s._b2
    n2 = norm(z2_pre, axis=-1, keepdims=true) + 1e-30
    z_p = z2_pre / n2
    # InfoNCE forward
    sim = z_a @ z_p.T / s.temperature
    sim -= sim.max(axis=-1, keepdims=true)
    exp_sim = exp(sim)
    row_sums = exp_sim.sum(axis=-1) + 1e-30
    n_batch = anchors.shape[0]
    pos_sim = collect([exp_sim[i, i] for i in 1:n_batch])
    loss = -mean(log(pos_sim / row_sums + 1e-30))
    cache = {
        "anchors": anchors,
        "positives": positives,
        "h1_pre": h1_pre,
        "h1": h1,
        "z1_pre": z1_pre,
        "n1": n1,
        "z_a": z_a,
        "h2_pre": h2_pre,
        "h2": h2,
        "z2_pre": z2_pre,
        "n2": n2,
        "z_p": z_p,
        "exp_sim": exp_sim,
        "row_sums": row_sums,
    }
    return float(loss), cache
end

function _backward(s::CEBRAEncoderState, cache, np.ndarray])
    n = cache["z_a"].shape[0]
    tau = s.temperature
    # dL/d(sim_matrix): softmax cross-entropy gradient
    probs = cache["exp_sim"] / cache["row_sums"][:, np.newaxis]
    d_sim = probs / n
    for i in 1:n
        d_sim[i, i] -= 1.0 / n
    # dL/dz_a, dL/dz_p from sim = z_a @ z_p.T / τ
    d_za = d_sim @ cache["z_p"] / tau
    d_zp = d_sim.T @ cache["z_a"] / tau
    # Backprop through L2 normalisation: z = z_pre / ||z_pre||
        z_hat = z_pre / norms
        return (d_z - z_hat * (d_z * z_hat).sum(axis=-1, keepdims=true)) / norms
    d_z1_pre = grad_l2norm(d_za, cache["z1_pre"], cache["n1"])
    d_z2_pre = grad_l2norm(d_zp, cache["z2_pre"], cache["n2"])
    # Backprop through layer 2 (both anchor && positive paths share weights)
    d_w2 = d_z1_pre.T @ cache["h1"] + d_z2_pre.T @ cache["h2"]
    d_b2 = d_z1_pre.sum(axis=0) + d_z2_pre.sum(axis=0)
    d_h1 = d_z1_pre @ s._w2
    d_h2 = d_z2_pre @ s._w2
    # ReLU gradient
    d_h1_pre = d_h1 * (cache["h1_pre"] > 0).astype(np.float64)
    d_h2_pre = d_h2 * (cache["h2_pre"] > 0).astype(np.float64)
    # Backprop through layer 1
    d_w1 = d_h1_pre.T @ cache["anchors"] + d_h2_pre.T @ cache["positives"]
    d_b1 = d_h1_pre.sum(axis=0) + d_h2_pre.sum(axis=0)
    return {"w1": d_w1, "b1": d_b1, "w2": d_w2, "b2": d_b2}
end

function fit(s::CEBRAEncoderState)
    self,
    data: np.ndarray,
    n_steps: int = 200,
    time_offset: int = 1,
    ) -> float
    n = data.shape[0] - time_offset
    if n < 2
        return 0.0
    anchors = data[:n]
    positives = data[time_offset : n + time_offset]
    loss = 0.0
    for _ in 1:n_steps
        loss, cache = s._forward_and_loss(anchors, positives)
        grads = s._backward(cache)
        s._w1 -= s.learning_rate * grads["w1"]
        s._b1 -= s.learning_rate * grads["b1"]
        s._w2 -= s.learning_rate * grads["w2"]
        s._b2 -= s.learning_rate * grads["b2"]
    return loss
end

function transform(s::CEBRAEncoderState, data)
    return s.encode(data)
end

end # module NeuralDecodersAccel
