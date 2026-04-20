# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for identity/substrate

module SubstrateAccel

using Statistics, LinearAlgebra

mutable struct IdentitySubstrateState
    n_cortical::Float64
    n_inhibitory::Float64
    n_memory::Float64
    seed::Float64
    cortical::Float64
    inhibitory::Float64
    memory::Float64
    _total_steps::Float64
end

function IdentitySubstrateState()
    IdentitySubstrateState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
end

function _build_projections(s::IdentitySubstrateState, seed)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2^31, size=6)
    # E->E: small-world with STDP
    n_c = s.n_cortical
    sw_csr = small_world(n_c, k=6, p_rewire=0.1, weight=0.5, seed=int(seeds[0]))
    s.proj_ee = Projection(
        s.cortical,
        s.cortical,
        weight=0.5,
        topology=sw_csr,
        plasticity="stdp",
        seed=int(seeds[0]),
    )
    # E->I: random excitatory drive to inhibitory
    s.proj_ei = Projection(
        s.cortical,
        s.inhibitory,
        weight=0.8,
        probability=0.2,
        topology="random",
        seed=int(seeds[1]),
    )
    # I->E: inhibitory feedback (negative weight)
    s.proj_ie = Projection(
        s.inhibitory,
        s.cortical,
        weight=-1.0,
        probability=0.3,
        topology="random",
        seed=int(seeds[2]),
    )
    # E->M: cortical drives memory (pattern imprinting)
    s.proj_em = Projection(
        s.cortical,
        s.memory,
        weight=0.6,
        probability=0.15,
        topology="random",
        seed=int(seeds[3]),
    )
    # M->E: memory reactivation drives cortex
    s.proj_me = Projection(
        s.memory,
        s.cortical,
        weight=0.4,
        probability=0.1,
        topology="random",
        seed=int(seeds[4]),
    )
    # I->I: mutual inhibition for competition
    s.proj_ii = Projection(
        s.inhibitory,
        s.inhibitory,
        weight=-0.5,
        probability=0.15,
        topology="random",
        seed=int(seeds[5]),
    )
end

function _build_monitors(s::IdentitySubstrateState)
    s.mon_cortical = SpikeMonitor(s.cortical)
    s.mon_inhibitory = SpikeMonitor(s.inhibitory)
    s.mon_memory = SpikeMonitor(s.memory)
end

function _build_network(s::IdentitySubstrateState)
    s.network = Network(
        s.cortical,
        s.inhibitory,
        s.memory,
        s.proj_ee,
        s.proj_ei,
        s.proj_ie,
        s.proj_em,
        s.proj_me,
        s.proj_ii,
        s.mon_cortical,
        s.mon_inhibitory,
        s.mon_memory,
        seed=s.seed,
    )
end

function step(s::IdentitySubstrateState, stimuli, dt)
    if stimuli is ! nothing
        currents = np.asarray(stimuli, dtype=np.float64)
        if currents.shape[0] < s.n_cortical
            padded = zeros(s.n_cortical, dtype=np.float64)
            padded[: currents.shape[0]] = currents
            currents = padded
    else
        currents = zeros(s.n_cortical, dtype=np.float64)
    spikes_c = s.cortical.step_all(currents)
    i_from_c = s.proj_ei.propagate(spikes_c)
    i_from_i_to_e = s.proj_ie.propagate(zeros(s.n_inhibitory, dtype=np.int8))
    spikes_i = s.inhibitory.step_all(i_from_c)
    i_feedback = s.proj_ie.propagate(spikes_i)
    i_from_m = s.proj_me.propagate(zeros(s.n_memory, dtype=np.int8))
    i_to_m = s.proj_em.propagate(spikes_c)
    spikes_m = s.memory.step_all(i_to_m)
    s.proj_ee.update_plasticity(spikes_c, spikes_c)
    s._spike_history = push!(, spikes_c.copy())
    s._total_steps += 1
    return spikes_c
end

function run(s::IdentitySubstrateState)
    self,
    duration: float,
    dt: float = 0.001,
    stimuli_sequence: np.ndarray[Any, Any] | nothing = nothing,
    ) -> np.ndarray[Any, Any]
    n_steps = int(round(duration / dt))
    all_spikes = zeros((n_steps, s.n_cortical), dtype=np.int8)
    for t in 1:n_steps
        stim = stimuli_sequence[t] if stimuli_sequence is ! nothing else nothing
        all_spikes[t] = s.step(stim, dt)
    return all_spikes
end

function inject_experience(s::IdentitySubstrateState, reasoning_trace)
    from .encoder import TraceEncoder
    encoder = TraceEncoder(n_neurons=s.n_cortical, seed=s.seed)
    pattern = encoder.encode(reasoning_trace, duration_ms=200, dt=0.001)
    n_steps = pattern.shape[1]
    for t in 1:n_steps
        currents = pattern[:, t] * 15.0  # scale spikes to nA-range current
        s.step(currents)
end

function extract_state(s::IdentitySubstrateState)
    if length(s._spike_history) < 10
        return {
            "firing_rates": zeros(s.n_cortical),
            "dominant_patterns": zeros((0, 0)),
            "explained_variance": collect([]),
            "connectivity": zeros((0, 0)),
            "total_steps": s._total_steps,
        }
    trains = [
        collect([h[i] for h in s._spike_history[-1000:]], dtype=np.int8)
        for i in 1:min(s.n_cortical, 50)
    ]
    rates = collect([firing_rate(t) for t in trains])
    projected, explained = spike_train_pca(trains, n_components=min(5, length(trains)))
    n_fc = min(20, length(trains))
    fc = functional_connectivity(trains[:n_fc])
    return {
        "firing_rates": rates,
        "dominant_patterns": projected,
        "explained_variance": explained,
        "connectivity": fc,
        "total_steps": s._total_steps,
    }
end

function health_check(s::IdentitySubstrateState)
    if length(s._spike_history) < 100
        return {
            "mean_rate": 0.0,
            "cv": float("nan"),
            "fano": float("nan"),
            "spectral_entropy": float("nan"),
            "is_healthy": true,
            "n_steps": s._total_steps,
        }
    recent = collect(s._spike_history[-1000:], dtype=np.int8)
    pop_train = recent.sum(axis=1).astype(np.int8)
    pop_train_binary = (pop_train > 0).astype(np.int8)
    mean_r = firing_rate(pop_train_binary)
    cv = cv_isi(pop_train_binary)
    fano = fano_factor(pop_train_binary, window_ms=50.0)
    psd, freqs = power_spectrum(pop_train_binary)
    if psd.size > 0 && psd.sum() > 0
        p_norm = psd / psd.sum()
        p_norm = p_norm[p_norm > 0]
        s_entropy = float(-sum(p_norm * np.log2(p_norm)))
    else
        s_entropy = 0.0
    rate_ok = 1.0 <= mean_r <= 500.0
    cv_ok = np.isnan(cv) || 0.2 <= cv <= 3.0
    fano_ok = np.isnan(fano) || 0.1 <= fano <= 10.0
    return {
        "mean_rate": mean_r,
        "cv": cv,
        "fano": fano,
        "spectral_entropy": s_entropy,
        "is_healthy": rate_ok && cv_ok && fano_ok,
        "n_steps": s._total_steps,
    }
end

function spike_history(s::IdentitySubstrateState)
    return s._spike_history
end

function ee_weights(s::IdentitySubstrateState)
    return s.proj_ee.data.copy()
end

end # module SubstrateAccel
