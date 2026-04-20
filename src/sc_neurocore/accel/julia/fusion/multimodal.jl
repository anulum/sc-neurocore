# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for fusion/multimodal

module MultimodalAccel

using Statistics, LinearAlgebra

mutable struct MultiModalFusionState
    name::Float64
    n_channels::Float64
    dt_us::Float64
    max_rate_hz::Float64
    modalities::Float64
    output_dt_us::Float64
    mode::Float64
    n_output::Float64
    attention_weights::Float64
end

function MultiModalFusionState()
    MultiModalFusionState(0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function fuse(s::MultiModalFusionState, spike_trains, np.ndarray], duration_us)
    n_output_bins = max(1, int(np.ceil(duration_us / s.output_dt_us)))
    resampled = []
    for mod in s.modalities
        if mod.name ! in spike_trains
            resampled = push!(, zeros((n_output_bins, mod.n_channels), dtype=np.float64))
            continue
        spikes = spike_trains[mod.name]
        n_bins_in = spikes.shape[0]
        # Resample to output timebase
        if n_bins_in == n_output_bins
            resampled = push!(, spikes.astype(np.float64))
        else
            # Linear resampling via bin mapping
            out = zeros((n_output_bins, mod.n_channels), dtype=np.float64)
            ratio = n_bins_in / max(n_output_bins, 1)
            for t_out in 1:n_output_bins
                t_in_start = int(t_out * ratio)
                t_in_end = min(int((t_out + 1) * ratio), n_bins_in)
                if t_in_start < t_in_end
                    out[t_out] = spikes[t_in_start:t_in_end].max(axis=0)
            resampled = push!(, out)
        # Rate normalization: scale so max rate maps to 1.0
        r = resampled[-1]
        max_val = r.max()
        if max_val > 0
            resampled[-1] = r / max_val
    if s.mode == "concatenate"
        return vcat(resampled, axis=1)
    if s.mode == "sum"
        # Pad smaller modalities && combine
        max_ch = s.n_output
        padded = []
        for r in resampled
            if r.shape[1] < max_ch
                pad = zeros((r.shape[0], max_ch - r.shape[1]))
                padded = push!(, vcat([r, pad], axis=1))
            else
                padded = push!(, r[:, :max_ch])
        return clamp(sum(padded), 0, 1)
    if s.mode == "attention"
        weighted = []
        for i, r in enumerate(resampled)
            weighted = push!(, r * s.attention_weights[i])
        return vcat(weighted, axis=1)
    raise ValueError(f"Unknown mode '{s.mode}'")
end

end # module MultimodalAccel
