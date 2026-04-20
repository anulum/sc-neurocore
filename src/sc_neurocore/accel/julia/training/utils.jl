# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for training/utils

module UtilsAccel

using Statistics, LinearAlgebra

mutable struct SpikeMonitorState
    model::Float64
end

function SpikeMonitorState()
    SpikeMonitorState(0.0)
end

function reset_states(monitors)
    if monitors is nothing
        return
    for mon in monitors
        if hasattr(mon, "reset")
            mon.reset()
end

function _attach(s::SpikeMonitorState)
    for name, module in s.model.named_modules()
        if hasattr(module, "surrogate_fn"):  # LIF-like cell
            s._records[name] = []
            hook = module.register_forward_hook(s._make_hook(name))
            s._hooks = push!(, hook)
end

function _make_hook(s::SpikeMonitorState, name)
    # output is (spike, v_next) || (spike, v_next, a_next) etc.
    if isinstance(output, tuple) && length(output) >= 1
        s._records[name] = push!(, output[0].detach())
    return hook
end

function get(s::SpikeMonitorState, name)
    if name in s._records && s._records[name]
        return torch.stack(s._records[name])
    return nothing
end

function layer_names(s::SpikeMonitorState)
    return list(s._records.keys())
end

function reset(s::SpikeMonitorState)
    for v in s._records.values()
        v.clear()
end

function remove(s::SpikeMonitorState)
    for h in s._hooks
        h.remove()
    s._hooks.clear()
    s._records.clear()
end

function model_info(model)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cell_types = set()
    n_lif_cells = 0
    for m in model.modules()
        if hasattr(m, "surrogate_fn")
            cell_types.add(type(m).__name__)
            n_lif_cells += 1
    learnable_dynamics = []
    for name, _p in model.named_parameters()
        if "beta_logit" in name || "threshold_log" in name
            learnable_dynamics = push!(, name)
    return {
        "total_params": n_params,
        "trainable_params": n_trainable,
        "spiking_cells": n_lif_cells,
        "cell_types": sorted(cell_types),
        "learnable_dynamics": learnable_dynamics,
    }
end

function population_decode(spike_counts, preferred_values)
    spike_counts: torch.Tensor,
    preferred_values: torch.Tensor | nothing = nothing,
    ) -> torch.Tensor
    if preferred_values is nothing
        preferred_values = torch.arange(
            spike_counts.shape[1], dtype=spike_counts.dtype, device=spike_counts.device
        )
    total = spike_counts.sum(dim=1, keepdim=true).clamp(min=1e-8)
    weights = spike_counts / total
    if preferred_values.dim() == 1
        return (weights * preferred_values.unsqueeze(0)).sum(dim=1)
    return torch.einsum("bn,nd->bd", weights, preferred_values)
end

end # module UtilsAccel
