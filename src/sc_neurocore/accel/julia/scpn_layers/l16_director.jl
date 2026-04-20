# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for scpn_layers/l16_director

module L16DirectorAccel

using Statistics, LinearAlgebra

mutable struct L16_DirectorLayerState
    n_control_nodes::Float64
    bitstream_length::Float64
    kp::Float64
    ki::Float64
    veto_threshold::Float64
    target_gci::Float64
    integral_clamp::Float64
    meta_coupling::Float64
    will::Float64
    integral_error::Float64
    entropy_proxy::Float64
    veto_active::Float64
    h_rec::Float64
    time::Float64
end

function L16_DirectorLayerState()
    L16_DirectorLayerState(10.0, 1024.0, 2.0, 0.5, 0.8, 0.8, 5.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function step(s::L16_DirectorLayerState)
    self,
    dt: float,
    l15_input: Optional[Dict[str, Any]] = nothing,
    ) -> Dict[str, Any]
    s.time += dt
    n = s.params.n_control_nodes
    gci = 0.5
    if l15_input is ! nothing && "gci" in l15_input
        gci = l15_input["gci"]
    # PI controller
    error = s.params.target_gci - gci
    s.integral_error = np.clip(
        s.integral_error + error * dt,
        -s.params.integral_clamp,
        s.params.integral_clamp,
    )
    u = s.params.kp * error + s.params.ki * s.integral_error
    u = clamp(u, -1, 1)
    # Entropy proxy (inverse of coherence stability)
    s.entropy_proxy = 0.9 * s.entropy_proxy + 0.1 * (1.0 - gci)
    # Veto
    s.veto_active = s.entropy_proxy > s.params.veto_threshold
    # Lyapunov candidate
    s.h_rec = abs(error) + (1 - gci) + s.entropy_proxy
    # Will update
    d_will = 0.1 * gci - 0.2 * s.entropy_proxy + 0.05 * u
    s.will = clamp(s.will + d_will * dt, 0, 1)
    effective_will = s.will * (0.0 if s.veto_active else 1.0)
    rands = np.random.random((n, s.params.bitstream_length))
    output_bitstreams = (rands < effective_will[:, nothing]).astype(np.uint8)
    return {
        "will": s.will.copy(),
        "control_signal": float(u),
        "veto_active": s.veto_active,
        "h_rec": s.h_rec,
        "entropy_proxy": s.entropy_proxy,
        "output_bitstreams": output_bitstreams,
    }
end

function get_global_metric(s::L16_DirectorLayerState)
    return float(mean(s.will))
end

end # module L16DirectorAccel
