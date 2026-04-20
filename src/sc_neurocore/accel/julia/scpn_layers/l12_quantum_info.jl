# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for scpn_layers/l12_quantum_info

module L12QuantumInfoAccel

using Statistics, LinearAlgebra

mutable struct L12_QuantumInfoLayerState
    n_sites::Float64
    bitstream_length::Float64
    transport_rate::Float64
    dephasing_gamma::Float64
    morphic_coupling::Float64
    coherence::Float64
    time::Float64
end

function L12_QuantumInfoLayerState()
    L12_QuantumInfoLayerState(100.0, 1024.0, 0.3, 0.05, 0.1, 0.0, 0.0)
end

function step(s::L12_QuantumInfoLayerState)
    self,
    dt: float,
    l11_input: Optional[Dict[str, Any]] = nothing,
    ) -> Dict[str, Any]
    s.time += dt
    n = s.params.n_sites
    # Nearest-neighbour transport (ring topology)
    transport = np.roll(s.coherence, 1) - 2 * s.coherence + np.roll(s.coherence, -1)
    dephasing = -s.params.dephasing_gamma * s.coherence
    s.coherence += (s.params.transport_rate * transport + dephasing) * dt
    if l11_input is ! nothing && "info_saturation" in l11_input
        s.coherence += 0.01 * l11_input["info_saturation"] * dt
    s.coherence = clamp(s.coherence, 0, 1)
    entropy = s._von_neumann_entropy()
    rands = np.random.random((n, s.params.bitstream_length))
    output_bitstreams = (rands < s.coherence[:, nothing]).astype(np.uint8)
    return {
        "coherence": s.coherence.copy(),
        "entropy": entropy,
        "transport_efficiency": float(mean(s.coherence)),
        "output_bitstreams": output_bitstreams,
    }
end

function _von_neumann_entropy(s::L12_QuantumInfoLayerState)
    p = s.coherence / (sum(s.coherence) + 1e-10)
    return float(-sum(p * log(p + 1e-10)))
end

function get_global_metric(s::L12_QuantumInfoLayerState)
    return float(mean(s.coherence))
end

end # module L12QuantumInfoAccel
