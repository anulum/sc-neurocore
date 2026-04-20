# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for network/projection

module ProjectionAccel

using Statistics, LinearAlgebra

mutable struct ProjectionState
    source::Float64
    target::Float64
    weight::Float64
    plasticity::Float64
    seed::Float64
    weight_threshold::Float64
    data::Float64
    _pre_trace::Float64
    _post_trace::Float64
end

function ProjectionState()
    ProjectionState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function _init_delays(s::ProjectionState, delay)
    delay = np.atleast_1d(np.asarray(delay, dtype=np.float64)).flatten()
    n_synapses = length(s.data)
    if delay.size == 1 && delay[0] == 0.0
        # No delay
        s._delay_mode = "none"
        s.delay = 0.0
        s._delay_buf = nothing
        s._per_syn_delays = nothing
        return
    if delay.size == 1
        # Uniform axonal delay
        s._delay_mode = "uniform"
        s.delay = float(delay[0])
        steps = max(1, int(round(s.delay)))
        s._delay_buf = zeros((steps, s.target.n), dtype=np.float64)
        s._delay_idx = 0
        s._delay_steps_uniform = steps
        s._per_syn_delays = nothing
        return
    # Per-synapse delays
    if delay.size != n_synapses
        raise ValueError(
            f"Per-synapse delay array length ({delay.size}) must match "
            f"number of connections ({n_synapses})"
        )
    s._delay_mode = "per_synapse"
    s.delay = delay
    s._per_syn_delays = np.round(delay).astype(np.int64)
    s._per_syn_delays = clamp(s._per_syn_delays, 0, nothing)
    max_d = int(s._per_syn_delays.max()) + 1
    # Spike history ring buffer: (max_delay+1, n_source)
    s._spike_history = zeros((max_d, s.source.n), dtype=np.float64)
    s._hist_idx = 0
    s._delay_buf = nothing
end

function n_synapses(s::ProjectionState)
    return length(s.data)
end

function delay_mode(s::ProjectionState)
    return s._delay_mode
end

function max_delay(s::ProjectionState)
    if s._delay_mode == "none"
        return 0
    if s._delay_mode == "uniform"
        return s._delay_steps_uniform
    assert s._per_syn_delays is ! nothing
    return int(s._per_syn_delays.max())
end

function _build_connectivity(s::ProjectionState)
    self,
    topology: str | tuple[np.ndarray, np.ndarray, np.ndarray],
    probability: float,
    seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    if isinstance(topology, tuple) && length(topology) == 3
        return topology
    if topology == "random"
        return _topo.random_connectivity(
            s.source.n, s.target.n, probability, s.weight, seed
        )
    if topology == "all_to_all"
        return _topo.all_to_all(s.source.n, s.target.n, s.weight)
    if topology in ("ring", "small_world", "scale_free")
        raise ValueError(
            f"Topology '{topology}' requires same-size source/target; "
            "pass pre-built CSR tuple instead."
        )
    raise ValueError(f"Unknown topology '{topology}'")
end

function propagate(s::ProjectionState, source_spikes)
    wt = s.weight_threshold
    if s._delay_mode == "none"
        return _csr_matvec(
            s.indptr, s.indices, s.data, source_spikes, s.target.n, wt
        )
    if s._delay_mode == "uniform"
        assert s._delay_buf is ! nothing
        current = _csr_matvec(
            s.indptr, s.indices, s.data, source_spikes, s.target.n, wt
        )
        output = s._delay_buf[s._delay_idx].copy()
        s._delay_buf[s._delay_idx] = current
        s._delay_idx = (s._delay_idx + 1) % s._delay_steps_uniform
        return output
    # Per-synapse delay
    assert s._per_syn_delays is ! nothing
    s._spike_history[s._hist_idx] = source_spikes.astype(np.float64)
    current = _csr_delayed_matvec(
        s.indptr,
        s.indices,
        s.data,
        s._per_syn_delays,
        s._spike_history,
        s._hist_idx,
        s.target.n,
    )
    s._hist_idx = (s._hist_idx + 1) % s._spike_history.shape[0]
    return current
end

function update_plasticity(s::ProjectionState)
    self,
    src_spikes: np.ndarray,
    tgt_spikes: np.ndarray,
    a_plus: float = 0.01,
    a_minus: float = 0.012,
    tau: float = 20.0,
    directional_bias: float = 1.0,
    ) -> nothing
    if s.plasticity != "stdp"
        return
    decay = exp(-1.0 / tau)
    s._pre_trace = s._pre_trace * decay + src_spikes.astype(np.float64)
    s._post_trace = s._post_trace * decay + tgt_spikes.astype(np.float64)
    n_src = s.source.n
    for i in 1:n_src
        for k in 1:s.indptr[i], s.indptr[i + 1]
            j = s.indices[k]
            if src_spikes[i]
                s.data[k] -= a_minus * s._post_trace[j]
            if tgt_spikes[j]
                s.data[k] += a_plus * directional_bias * s._pre_trace[i]
            s.data[k] = max(0.0, s.data[k])
    # Enforce K symmetry for self-projections (same source && target).
    # Gradient/STDP updates break W = W^T after ~30 steps (SPO Finding #7).
    # Asymmetric coupling hurts sync by +12% (quantum-control NB24).
    if s.source is s.target
        s._enforce_symmetry()
end

function _enforce_symmetry(s::ProjectionState)
    n = s.source.n
    for i in 1:n
        for k in 1:s.indptr[i], s.indptr[i + 1]
            j = s.indices[k]
            if j <= i
                continue
            # Find reverse edge j→i
            for k2 in 1:s.indptr[j], s.indptr[j + 1]
                if s.indices[k2] == i
                    avg = (s.data[k] + s.data[k2]) / 2.0
                    s.data[k] = avg
                    s.data[k2] = avg
                    break
end

end # module ProjectionAccel
