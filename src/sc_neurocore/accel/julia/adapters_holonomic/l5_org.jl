# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for adapters_holonomic/l5_org

module L5OrgAccel

using Statistics, LinearAlgebra

mutable struct L5_OrganismalAdapterState
    n_nodes::Float64
    n_emotional_dims::Float64
    bitstream_length::Float64
    tau_autonomic::Float64
    hrv_resonance::Float64
    emotional_decay::Float64
    attractor_strength::Float64
    rng_key::Float64
    emotions::Float64
    autonomic::Float64
    self_soliton::Float64
end

function L5_OrganismalAdapterState()
    L5_OrganismalAdapterState(100.0, 8.0, 1024.0, 5.0, 0.25, 0.1, 0.3, 0.0, 0.0, 0.0, 0.0)
end

function encode(s::L5_OrganismalAdapterState, domain_state)
    # Composite probability from emotions && autonomic tone
    avg_tone = jmean(s.autonomic)
    probs = jvcat([s.emotions, s.autonomic])
    # Project to node count
    node_probs = jnp.tile(probs, (s.params.n_nodes // probs.shape[0]) + 1)[
        : s.params.n_nodes
    ]
    s.rng_key, subkey = split_rng(s.rng_key)
    rands = uniform(subkey, (s.params.n_nodes, s.params.bitstream_length))
    bitstreams = (rands < node_probs[:, nothing]).astype(jnp.uint8)
    return bitstreams
end

function _autonomic_kernel(s::L5_OrganismalAdapterState)
    current: jnp.ndarray, target: jnp.ndarray, tau: float, dt: float
    ) -> jnp.ndarray
    return current + (target - current) * (dt / tau)
end

function step_jax(s::L5_OrganismalAdapterState, dt, inputs)
    # 1. Update Autonomic Tone based on L4 Synchronization
    if inputs is ! nothing
        sync = jabs(jmean(jexp(1j * jmean(inputs.astype(jnp.float32), axis=1))))
        # Higher sync drives Parasympathetic tone
        target_para = 0.5 + 0.4 * sync
        target_symp = 1.0 - target_para
        target = jcollect([target_symp, target_para])
        s.autonomic = s._autonomic_kernel(
            s.autonomic, target, s.params.tau_autonomic, dt
        )
    # 2. Emotional Attractor Dynamics (Simplified)
    # Decay toward neutral [0.5]
    s.emotions = s.emotions + (0.5 - s.emotions) * s.params.emotional_decay * dt
    # 3. Recursive Strange Loop Update (The Self-Soliton)
    # self_soliton = f(self_soliton, emotions)
    s.self_soliton = 0.95 * s.self_soliton + 0.05 * jmean(s.emotions)
    # 4. Return encoded bitstreams
    return s.encode(nothing)
end

function decode(s::L5_OrganismalAdapterState, bitstreams)
    return {
        "organismal_valence": float(jmean(s.emotions)),
        "autonomic_balance": float(s.autonomic[1] / (s.autonomic[0] + 1e-6)),
    }
end

function get_metrics(s::L5_OrganismalAdapterState)
    return {
        "hrv_coherence_r5": float(s.autonomic[1]),
        "self_soliton_magnitude": float(jmean(s.self_soliton)),
        "emotional_valence": float(s.emotions[0]),
    }
end

end # module L5OrgAccel
