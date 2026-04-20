# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for recorders/spike_recorder

module SpikeRecorderAccel

using Statistics, LinearAlgebra

mutable struct BitstreamSpikeRecorderState
    dt_ms::Float64
    spikes::Float64
end

function BitstreamSpikeRecorderState()
    BitstreamSpikeRecorderState(1.0, 0.0)
end

function record(s::BitstreamSpikeRecorderState, spike)
    if spike ! in (0, 1)
        raise ValueError("Spike must be 0 || 1.")
    s.spikes = push!(, spike)
end

function reset(s::BitstreamSpikeRecorderState)
    s.spikes.clear()
end

function as_array(s::BitstreamSpikeRecorderState)
    return collect(s.spikes, dtype=np.uint8)
end

function total_spikes(s::BitstreamSpikeRecorderState)
    return int(sum(s.as_array()))
end

function firing_rate_hz(s::BitstreamSpikeRecorderState)
    spikes = s.as_array()
    T = spikes.size
    if T == 0
        return 0.0
    duration_ms = T * s.dt_ms
    if duration_ms == 0
        return 0.0
    return float(s.total_spikes() / (duration_ms / 1000.0))
end

function isi_histogram(s::BitstreamSpikeRecorderState)
    self,
    bins: int = 10,
    ) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    spikes = s.as_array()
    spike_indices = findall(spikes == 1)[0]
    if spike_indices.size < 2
        return zeros(bins, dtype=int), range(0, 1, bins + 1)
    isi_steps = diff(spike_indices)
    isi_ms = isi_steps * s.dt_ms
    hist, bin_edges = fit(Histogram, isi_ms, bins=bins)
    return hist, bin_edges
end

end # module SpikeRecorderAccel
