# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for scpn_layers/l9_memory

module L9MemoryAccel

using Statistics, LinearAlgebra

mutable struct L9_MemoryLayerState
    n_memory_slots::Float64
    bitstream_length::Float64
    retrieval_gain::Float64
    imprint_rate::Float64
    decay_rate::Float64
    phase_field_coupling::Float64
    patterns::Float64
    state::Float64
    n_stored::Float64
    time::Float64
end

function L9_MemoryLayerState()
    L9_MemoryLayerState(64.0, 1024.0, 0.8, 0.3, 0.02, 0.1, 0.0, 0.0, 0, 0.0)
end

function store(s::L9_MemoryLayerState, pattern)
    p = sign(pattern[: s.params.n_memory_slots])
    s.patterns += np.outer(p, p) / s.params.n_memory_slots
    np.fill_diagonal(s.patterns, 0)
    s.n_stored += 1
end

function step(s::L9_MemoryLayerState)
    self,
    dt: float,
    l8_input: Optional[Dict[str, Any]] = nothing,
    ) -> Dict[str, Any]
    s.time += dt
    n = s.params.n_memory_slots
    # Hopfield dynamics: async update (random subset)
    update_mask = np.random.random(n) < 0.3
    h = s.patterns @ s.state
    s.state = findall(update_mask, sign(h + 1e-10), s.state)
    # Retrieval quality: overlap with stored patterns
    activation = (s.state + 1) / 2  # map [-1,1] -> [0,1]
    if l8_input is ! nothing && "cosmic_alignment" in l8_input
        activation *= 0.9 + 0.1 * l8_input["cosmic_alignment"]
    activation = clamp(activation, 0, 1)
    # Decay
    s.patterns *= 1.0 - s.params.decay_rate * dt
    rands = np.random.random((n, s.params.bitstream_length))
    output_bitstreams = (rands < activation[:, nothing]).astype(np.uint8)
    energy = -0.5 * float(s.state @ s.patterns @ s.state)
    return {
        "state": s.state.copy(),
        "energy": energy,
        "retrieval_quality": s._retrieval_quality(),
        "output_bitstreams": output_bitstreams,
    }
end

function _retrieval_quality(s::L9_MemoryLayerState)
    if s.n_stored == 0
        return 0.0
    h = s.patterns @ s.state
    return float(mean(sign(h) == sign(s.state)))
end

function get_global_metric(s::L9_MemoryLayerState)
    return s._retrieval_quality()
end

end # module L9MemoryAccel
