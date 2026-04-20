# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for scpn_layers/l11_morphic

module L11MorphicAccel

using Statistics, LinearAlgebra

mutable struct L11_MorphicLayerState
    n_nodes::Float64
    bitstream_length::Float64
    j_coupling::Float64
    h_bias::Float64
    beta_infection::Float64
    gamma_recovery::Float64
    boundary_coupling::Float64
    spins::Float64
    info_density::Float64
    time::Float64
end

function L11_MorphicLayerState()
    L11_MorphicLayerState(100.0, 1024.0, 0.5, 0.1, 0.2, 0.05, 0.1, 0.0, 0.0, 0.0)
end

function step(s::L11_MorphicLayerState)
    self,
    dt: float,
    l10_input: Optional[Dict[str, Any]] = nothing,
    ) -> Dict[str, Any]
    s.time += dt
    n = s.params.n_nodes
    field_input = zeros(n)
    if l10_input is ! nothing && "integrity" in l10_input
        field_input = np.full(n, l10_input["integrity"] * 0.1)
    mean_field = mean(s.spins)
    d_spin = (
        s.params.j_coupling * mean_field
        + s.params.h_bias
        + field_input
        - 0.1 * s.spins
    )
    s.spins = clamp(s.spins + d_spin * dt, 0, 1)
    s.info_density = 0.9 * s.info_density + 0.1 * abs(s.spins - 0.5)  # type: ignore[assignment]
    rands = np.random.random((n, s.params.bitstream_length))
    output_bitstreams = (rands < s.spins[:, nothing]).astype(np.uint8)
    return {
        "spins": s.spins.copy(),
        "polarization": float(std(s.spins)),
        "info_saturation": float(mean(s.info_density)),
        "output_bitstreams": output_bitstreams,
    }
end

function get_global_metric(s::L11_MorphicLayerState)
    return float(mean(s.spins))
end

end # module L11MorphicAccel
