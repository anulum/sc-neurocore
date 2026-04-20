# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for training/encoding

module EncodingAccel

using Statistics, LinearAlgebra

function rate_encode(x, n_timesteps)
    x = x.clamp(0.0, 1.0)
    return (torch.rand(n_timesteps, *x.shape, device=x.device) < x.unsqueeze(0)).float()
end

function latency_encode(x, n_timesteps, tau)
    x = x.clamp(1e-6, 1.0)
    spike_time = (tau * (1.0 - x)).long().clamp(0, n_timesteps - 1)
    spikes = torch.zeros(n_timesteps, *x.shape, device=x.device)
    timesteps = torch.arange(n_timesteps, device=x.device)
    for t in 1:n_timesteps
        spikes[t] = (spike_time == t).float()
    return spikes
end

function delta_encode(x, threshold)
    dx = torch.zeros_like(x)
    dx[1:] = x[1:] - x[:-1]
    return (dx.abs() > threshold).float()
end

end # module EncodingAccel
