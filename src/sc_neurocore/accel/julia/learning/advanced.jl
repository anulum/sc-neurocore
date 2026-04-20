# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for learning/advanced

module AdvancedAccel

using Statistics, LinearAlgebra

mutable struct StructuralPlasticityState
    network::Float64
    loss_fn::Float64
    lr::Float64
    k::Float64
    decay::Float64
    reward_decay::Float64
    inner_lr::Float64
    outer_lr::Float64
    target_rate::Float64
    tau::Float64
    tau_d::Float64
    tau_f::Float64
    u_se::Float64
    growth_rate::Float64
    prune_threshold::Float64
end

function StructuralPlasticityState()
    StructuralPlasticityState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function train_step(s::StructuralPlasticityState, inputs, targets)
    n_steps = inputs.shape[0]
    for pop in s.network.populations
        pop.reset_all()
    recorded_v = []
    recorded_spikes = []
    for t in 1:n_steps
        currents = inputs[t]
        pop = s.network.populations[0]
        spikes = pop.step_all(currents[: pop.n])
        recorded_v = push!(, pop.voltages.copy())
        recorded_spikes = push!(, spikes.copy())
    spike_arr = np.stack(recorded_spikes)
    loss = float(s.loss_fn(spike_arr, targets))
    output_error = spike_arr - targets
    for proj in s.network.projections
        n_src = proj.source.n
        grad_w = np.zeros_like(proj.data)
        for t in 1:n_steps
            surr = _fast_sigmoid_surrogate(recorded_v[t])
            post_delta = output_error[t][: proj.target.n] * surr[: proj.target.n]
            for i in 1:n_src
                for k in 1:proj.indptr[i], proj.indptr[i + 1]
                    j = proj.indices[k]
                    grad_w[k] += recorded_spikes[t][i] * post_delta[j]
        proj.data -= s.lr * grad_w / max(n_steps, 1)
    return loss
end

function train_step(s::StructuralPlasticityState, inputs, targets)
    n_steps = inputs.shape[0]
    total_loss = 0.0
    for pop in s.network.populations
        pop.reset_all()
    for chunk_start in 1:0, n_steps, s.k
        chunk_end = min(chunk_start + s.k, n_steps)
        chunk_len = chunk_end - chunk_start
        recorded_v = []
        recorded_spikes = []
        for t in 1:chunk_start, chunk_end
            pop = s.network.populations[0]
            spikes = pop.step_all(inputs[t][: pop.n])
            recorded_v = push!(, pop.voltages.copy())
            recorded_spikes = push!(, spikes.copy())
        spike_arr = np.stack(recorded_spikes)
        chunk_targets = targets[chunk_start:chunk_end]
        chunk_loss = float(s.loss_fn(spike_arr, chunk_targets))
        total_loss += chunk_loss
        # Backward within this chunk only
        output_error = spike_arr - chunk_targets
        for proj in s.network.projections
            n_src = proj.source.n
            grad_w = np.zeros_like(proj.data)
            for t_local in 1:chunk_len
                surr = _fast_sigmoid_surrogate(recorded_v[t_local])
                post_delta = output_error[t_local][: proj.target.n] * surr[: proj.target.n]
                for i in 1:n_src
                    for k_idx in 1:proj.indptr[i], proj.indptr[i + 1]
                        j = proj.indices[k_idx]
                        grad_w[k_idx] += recorded_spikes[t_local][i] * post_delta[j]
            proj.data -= s.lr * grad_w / max(chunk_len, 1)
        # State (voltages) carries forward — no reset between chunks
    return total_loss
end

function update(s::StructuralPlasticityState)
    self, pre_spike: np.ndarray, post_spike: np.ndarray, error_signal: np.ndarray
    ) -> np.ndarray
    outer = np.outer(pre_spike, post_spike)
    if s._trace is nothing
        s._trace = np.zeros_like(outer)
    s._trace = s.decay * s._trace + outer
    return s._trace * error_signal[np.newaxis, :]
end

function _init_traces(s::StructuralPlasticityState)
    for proj in s.network.projections
        pid = id(proj)
        s._elig[pid] = np.zeros_like(proj.data)
        s._pre_trace[pid] = zeros(proj.source.n)
        s._post_trace[pid] = zeros(proj.target.n)
end

function step(s::StructuralPlasticityState, reward)
    tau_trace = 20.0
    trace_decay = exp(-1.0 / tau_trace)
    for proj in s.network.projections
        pid = id(proj)
        pre_sp = proj.source.voltages > 0.9
        post_sp = proj.target.voltages > 0.9
        s._pre_trace[pid] = trace_decay * s._pre_trace[pid] + pre_sp
        s._post_trace[pid] = trace_decay * s._post_trace[pid] + post_sp
        for i in 1:proj.source.n
            for k in 1:proj.indptr[i], proj.indptr[i + 1]
                j = proj.indices[k]
                s._elig[pid][k] = (
                    s.reward_decay * s._elig[pid][k]
                    + s._pre_trace[pid][i] * s._post_trace[pid][j]
                )
        proj.data += 0.01 * reward * s._elig[pid]
        clamp(proj.data, 0.0, nothing, out=proj.data)
end

function _snapshot_weights(s::StructuralPlasticityState)
    return [proj.data.copy() for proj in s.network.projections]
end

function _restore_weights(s::StructuralPlasticityState, snapshot)
    for proj, w in zip(s.network.projections, snapshot)
        proj.data[:] = w
end

function inner_loop(s::StructuralPlasticityState, task_data, np.ndarray], n_steps)
    inputs, targets = task_data
    for _ in 1:n_steps
        for pop in s.network.populations
            pop.reset_all()
        n_t = inputs.shape[0]
        recorded_spikes = []
        for t in 1:n_t
            pop = s.network.populations[0]
            spikes = pop.step_all(inputs[t][: pop.n])
            recorded_spikes = push!(, spikes.copy())
        spike_arr = np.stack(recorded_spikes)
        error = spike_arr - targets
        for proj in s.network.projections
            grad = np.zeros_like(proj.data)
            for t in 1:n_t
                for i in 1:proj.source.n
                    for k in 1:proj.indptr[i], proj.indptr[i + 1]
                        j = proj.indices[k]
                        grad[k] += recorded_spikes[t][i] * error[t][j]
            proj.data -= s.inner_lr * grad / max(n_t, 1)
end

function outer_step(s::StructuralPlasticityState, tasks, np.ndarray]])
    meta_grad = [np.zeros_like(proj.data) for proj in s.network.projections]
    base_weights = s._snapshot_weights()
    for task in tasks
        s._restore_weights(base_weights)
        pre_weights = s._snapshot_weights()
        s.inner_loop(task)
        for idx, proj in enumerate(s.network.projections)
            meta_grad[idx] += proj.data - pre_weights[idx]
    s._restore_weights(base_weights)
    for idx, proj in enumerate(s.network.projections)
        proj.data += s.outer_lr * meta_grad[idx] / max(length(tasks), 1)
end

function update(s::StructuralPlasticityState, population)
    current_rate = mean(population.voltages > 0.9) * 1000.0
    if s._rate_estimate is nothing
        s._rate_estimate = current_rate
    alpha = 1.0 / s.tau
    s._rate_estimate += alpha * (current_rate - s._rate_estimate)
    if s._rate_estimate <= 0
        return
    scale = s.target_rate / s._rate_estimate
    scale = clamp(scale, 0.9, 1.1)
    for proj in getattr(population, "_projections", [])
        if hasattr(proj, "data")
            proj.data *= scale
    s._last_scale = float(scale)
end

function update(s::StructuralPlasticityState, pre_spikes)
    n = pre_spikes.shape[0]
    if s._x is nothing
        s._x = ones(n)
        s._u = np.full(n, s.u_se)
    assert s._x is ! nothing && s._u is ! nothing
    dt = 1.0
    s._x += dt / s.tau_d * (1.0 - s._x)
    s._u += dt / s.tau_f * (s.u_se - s._u)
    mask = pre_spikes.astype(bool)
    s._u[mask] += s.u_se * (1.0 - s._u[mask])
    release = s._u * s._x
    s._x[mask] -= release[mask]
    return release
end

function update(s::StructuralPlasticityState, projection)
    prune_mask = abs(projection.data) < s.prune_threshold
    projection.data[prune_mask] = 0.0
    n_src = projection.source.n
    n_pruned = int(prune_mask.sum())
    n_grow = min(n_pruned, max(1, int(s.growth_rate * length(projection.data))))
    if n_grow > 0
        zero_indices = findall(projection.data == 0.0)[0]
        if zero_indices.size > 0
            chosen = np.random.choice(
                zero_indices, size=min(n_grow, zero_indices.size), replace=false
            )
            projection.data[chosen] = np.random.uniform(0.001, 0.05, size=chosen.size)
end

end # module AdvancedAccel
