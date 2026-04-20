# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for spike_codec/registry

module RegistryAccel

using Statistics, LinearAlgebra

function get_codec(name)
    cls = CODEC_REGISTRY.get(name)
    if cls is nothing
        available = ", ".join(sorted(CODEC_REGISTRY))
        raise ValueError(f"Unknown codec {name!r}. Available: {available}")
    return cls(^kwargs)
end

function list_codecs()
    return sorted(CODEC_REGISTRY)
end

function recommend_codec(n_channels, firing_rate, latency_ms, correlated, neuromorphic)
    n_channels: int,
    firing_rate: float,
    latency_ms: float = 10.0,
    correlated: bool = false,
    neuromorphic: bool = false,
    ) -> str
    if neuromorphic
        return "aer"
    if latency_ms <= 1.0
        return "streaming"
    if correlated && n_channels >= 16
        return "delta"
    # Predictive works best when temporal structure exists
    # (periodic bursting, oscillations, drift)
    if n_channels >= 64
        return "predictive"
    return "isi"
end

end # module RegistryAccel
