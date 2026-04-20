# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for scpn_layers/l10_boundary

module L10BoundaryAccel

using Statistics, LinearAlgebra

mutable struct L10_BoundaryLayerState
    n_boundary_nodes::Float64
    bitstream_length::Float64
    rejection_threshold::Float64
    shielding_strength::Float64
    steering_gain::Float64
    memory_coupling::Float64
    firewall_strength::Float64
    intention::Float64
    time::Float64
end

function L10_BoundaryLayerState()
    L10_BoundaryLayerState(100.0, 1024.0, 0.4, 1.5, 0.2, 0.1, 0.0, 0.0, 0.0)
end

function step(s::L10_BoundaryLayerState)
    self,
    dt: float,
    l9_input: Optional[Dict[str, Any]] = nothing,
    external_noise: Optional[np.ndarray] = nothing,
    ) -> Dict[str, Any]
    s.time += dt
    n = s.params.n_boundary_nodes
    noise = zeros(n)
    if external_noise is ! nothing
        noise = (
            external_noise[:n]  # type: ignore[assignment]
            if length(external_noise) >= n
            else np.pad(external_noise, (0, n - length(external_noise)))
        )
    if l9_input is ! nothing && "retrieval_quality" in l9_input
        s.intention = np.full(n, l9_input["retrieval_quality"])
    dissonance = abs(noise - s.intention)
    d_strength = (
        -dissonance * s.firewall_strength
        + s.params.steering_gain * s.intention
        - 0.01 * s.firewall_strength
    )
    s.firewall_strength = clamp(s.firewall_strength + d_strength * dt, 0, 1)
    rands = np.random.random((n, s.params.bitstream_length))
    output_bitstreams = (rands < s.firewall_strength[:, nothing]).astype(np.uint8)
    return {
        "firewall_strength": s.firewall_strength.copy(),
        "dissonance": float(mean(dissonance)),
        "integrity": s._integrity(),
        "output_bitstreams": output_bitstreams,
    }
end

function _integrity(s::L10_BoundaryLayerState)
    return float(mean(s.firewall_strength))
end

function get_global_metric(s::L10_BoundaryLayerState)
    return s._integrity()
end

end # module L10BoundaryAccel
