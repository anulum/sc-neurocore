# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for adapters_holonomic/l9_mem

module L9MemAccel

using Statistics, LinearAlgebra

mutable struct L9_MemoryAdapterState
    n_memory_slots::Float64
    bitstream_length::Float64
    retrieval_gain::Float64
    weak_measurement_strength::Float64
    temporal_window::Float64
    rng_key::Float64
    imprints_psi::Float64
    retrieval_phi::Float64
    current_slot::Float64
end

function L9_MemoryAdapterState()
    L9_MemoryAdapterState(64.0, 1024.0, 0.8, 0.1, 100.0, 0.0, 0.0, 0.0, 0)
end

function encode(s::L9_MemoryAdapterState, domain_state)
    # Memory retrieval probability = Normalized overlap <Phi|Psi>
    psi_float = s.imprints_psi.astype(jnp.float32)
    phi_float = s.retrieval_phi.astype(jnp.float32)
    # Calculate overlap per slot
    overlap = jmean(psi_float * phi_float, axis=1)
    # Sum overlaps to get retrieval activation
    retrieval_prob = jclamp(jsum(overlap) * s.params.retrieval_gain, 0.0, 1.0)
    s.rng_key, subkey = split_rng(s.rng_key)
    rands = uniform(subkey, (s.params.bitstream_length,))
    # Single channel output representing retrieved memory content
    bitstream = (rands < retrieval_prob).astype(jnp.uint8)
    return bitstream
end

function _tsvf_kernel(s::L9_MemoryAdapterState)
    psi: jnp.ndarray, phi: jnp.ndarray, inputs: jnp.ndarray, strength: float, dt: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]
    # Forward imprinting Psi captures current input
    psi_next = jfindall(inputs > 0.5, 1, psi).astype(jnp.uint8)
    # Backward retrieval Phi adapts to current state (Weak measurement)
    phi_next = jfindall(jabs(psi_next.astype(jnp.float32) - 0.5) > 0.1, 1, phi).astype(
        jnp.uint8
    )
    return psi_next, phi_next
end

function step_jax(s::L9_MemoryAdapterState, dt, inputs)
    if inputs is ! nothing
        # 1. Project inputs to memory slot count if necessary
        if inputs.shape[0] != s.params.n_memory_slots
            # Tile || truncate to match slots
            n_in = inputs.shape[0]
            n_slots = s.params.n_memory_slots
            indices = jcollect(n_slots) % n_in
            mapped_inputs = inputs[indices]
        else
            mapped_inputs = inputs
        # 2. Update forward/backward holographic imprints
        s.imprints_psi, s.retrieval_phi = s._tsvf_kernel(
            s.imprints_psi,
            s.retrieval_phi,
            mapped_inputs,
            s.params.weak_measurement_strength,
            dt,
        )
    # 3. Return retrieved bitstream (projected to node count)
    return s.encode(nothing)
end

function decode(s::L9_MemoryAdapterState, bitstreams)
    return {"memory_retrieval_r9": float(jmean(bitstreams.astype(jnp.float32)))}
end

function get_metrics(s::L9_MemoryAdapterState)
    return {
        "holographic_overlap": float(
            jmean(
                s.imprints_psi.astype(jnp.float32) * s.retrieval_phi.astype(jnp.float32)
            )
        ),
        "imprint_density": float(jmean(s.imprints_psi)),
    }
end

end # module L9MemAccel
