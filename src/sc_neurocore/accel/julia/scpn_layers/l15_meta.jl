# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for scpn_layers/l15_meta

module L15MetaAccel

using Statistics, LinearAlgebra

mutable struct L15_MetaLayerState
    n_monitors::Float64
    bitstream_length::Float64
    target_coherence::Float64
    smoothing_alpha::Float64
    integration_coupling::Float64
    gci::Float64
    error_history::Float64
    time::Float64
end

function L15_MetaLayerState()
    L15_MetaLayerState(16.0, 1024.0, 0.8, 0.1, 0.2, 0.5, 0.0, 0.0)
end

function step(s::L15_MetaLayerState)
    self,
    dt: float,
    l14_input: Optional[Dict[str, Any]] = nothing,
    ) -> Dict[str, Any]
    s.time += dt
    actual = 0.5
    if l14_input is ! nothing && "integrated_coherence" in l14_input
        actual = l14_input["integrated_coherence"]
    error = abs(s.params.target_coherence - actual)
    s.gci = (1 - s.params.smoothing_alpha) * s.gci + s.params.smoothing_alpha * (
        1 - error
    )
    # Per-monitor error tracking (shift && append)
    s.error_history = np.roll(s.error_history, -1)  # type: ignore[assignment]
    s.error_history[-1] = error
    activation = np.full(s.params.n_monitors, clamp(s.gci, 0, 1))
    rands = np.random.random((s.params.n_monitors, s.params.bitstream_length))
    output_bitstreams = (rands < activation[:, nothing]).astype(np.uint8)
    return {
        "gci": s.gci,
        "error": error,
        "error_trend": float(mean(s.error_history)),
        "output_bitstreams": output_bitstreams,
    }
end

function get_global_metric(s::L15_MetaLayerState)
    return s.gci
end

end # module L15MetaAccel
