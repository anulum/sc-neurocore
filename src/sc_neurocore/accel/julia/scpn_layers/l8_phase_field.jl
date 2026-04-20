# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for scpn_layers/l8_phase_field

module L8PhaseFieldAccel

using Statistics, LinearAlgebra

mutable struct L8_PhaseFieldLayerState
    n_pulsars::Float64
    bitstream_length::Float64
    k_cosmic::Float64
    symbolic_coupling::Float64
    director_coupling::Float64
    pulsar_omegas::Float64
    phases::Float64
    time::Float64
end

function L8_PhaseFieldLayerState()
    L8_PhaseFieldLayerState(12.0, 1024.0, 0.05, 0.1, 0.15, 0.0, 0.0, 0.0)
end

function step(s::L8_PhaseFieldLayerState)
    self,
    dt: float,
    l7_input: Optional[Dict[str, Any]] = nothing,
    ) -> Dict[str, Any]
    s.time += dt
    n = s.params.n_pulsars
    omegas = s.params.pulsar_omegas
    # Kuramoto coupling: phase differences
    phase_diff = s.phases[np.newaxis, :] - s.phases[:, np.newaxis]
    coupling = s.params.k_cosmic * sum(sin(phase_diff), axis=1) / n
    d_phase = omegas + coupling
    if l7_input is ! nothing && "glyph_vector" in l7_input
        drive = mean(l7_input["glyph_vector"])
        d_phase += s.params.symbolic_coupling * drive * sin(-s.phases)
    s.phases = (s.phases + d_phase * dt) % (2 * pi)
    activation = (1.0 + cos(s.phases)) / 2.0
    rands = np.random.random((n, s.params.bitstream_length))
    output_bitstreams = (rands < activation[:, nothing]).astype(np.uint8)
    return {
        "phases": s.phases.copy(),
        "cosmic_alignment": s._order_parameter(),
        "output_bitstreams": output_bitstreams,
    }
end

function _order_parameter(s::L8_PhaseFieldLayerState)
    return float(abs(mean(exp(1j * s.phases))))
end

function get_global_metric(s::L8_PhaseFieldLayerState)
    return s._order_parameter()
end

end # module L8PhaseFieldAccel
