# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for training/losses

module LossesAccel

using Statistics, LinearAlgebra

function spike_count_loss(spike_counts, targets)
    return F.cross_entropy(spike_counts, targets)
end

function membrane_loss(membrane_acc, targets)
    return F.cross_entropy(membrane_acc, targets)
end

function spike_rate_loss(spike_counts, targets, n_timesteps, target_rate)
    spike_counts: torch.Tensor,
    targets: torch.Tensor,
    n_timesteps: int,
    target_rate: float = 0.8,
    ) -> torch.Tensor
    rates = spike_counts / n_timesteps
    n_classes = rates.shape[1]
    bg_rate = (1.0 - target_rate) / max(n_classes - 1, 1)
    target_rates = torch.full_like(rates, bg_rate)
    target_rates.scatter_(1, targets.unsqueeze(1), target_rate)
    return F.mse_loss(rates, target_rates)
end

function spike_l1_loss(spike_counts, n_timesteps)
    return (spike_counts / n_timesteps).abs().mean()
end

function spike_l2_loss(spike_counts, n_timesteps)
    return ((spike_counts / n_timesteps) ^ 2).mean()
end

end # module LossesAccel
