# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for scpn_layers/l14_integration

module L14IntegrationAccel

using Statistics, LinearAlgebra

mutable struct L14_IntegrationLayerState
    n_dimensions::Float64
    bitstream_length::Float64
    integration_weights::Float64
    temporal_coupling::Float64
    layer_metrics::Float64
    integrated_coherence::Float64
    time::Float64
end

function L14_IntegrationLayerState()
    L14_IntegrationLayerState(13.0, 1024.0, 0.0, 0.1, 0.0, 0.5, 0.0)
end

function step(s::L14_IntegrationLayerState)
    self,
    dt: float,
    layer_metrics: Optional[Dict[str, float]] = nothing,
    ) -> Dict[str, Any]
    s.time += dt
    if layer_metrics is ! nothing
        values = list(layer_metrics.values())[: s.params.n_dimensions]
        s.layer_metrics[: length(values)] = values
    w = s.params.integration_weights
    s.integrated_coherence = float(dot(w, s.layer_metrics))  # type: ignore[arg-type]
    activation = np.full(s.params.n_dimensions, s.integrated_coherence)
    activation = clamp(activation, 0, 1)  # type: ignore[assignment]
    rands = np.random.random((s.params.n_dimensions, s.params.bitstream_length))
    output_bitstreams = (rands < activation[:, nothing]).astype(np.uint8)
    return {
        "integrated_coherence": s.integrated_coherence,
        "layer_metrics": s.layer_metrics.copy(),
        "output_bitstreams": output_bitstreams,
    }
end

function get_global_metric(s::L14_IntegrationLayerState)
    return s.integrated_coherence
end

end # module L14IntegrationAccel
