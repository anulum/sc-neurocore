# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for utils

fn reset_states(monitors: Int) -> Int:
    var _reset_states_line = 'if monitors is 0:'
    return 0  # return
    var _reset_states_line = 'for mon in monitors:'
    var _reset_states_line = 'if hasattr(mon, "reset"):'
    var _reset_states_line = 'mon.reset()'

fn model_info(model: Int) -> Int:
    var _model_info_line = 'n_params = sum(p.numel() for p in model.parameters())'
    var _model_info_line = 'n_trainable = sum(p.numel() for p in model.parameters() if p'
    var _model_info_line = 'cell_types = set()'
    var _model_info_line = 'n_lif_cells = 0'
    var _model_info_line = 'for m in model.modules():'
    var _model_info_line = 'if hasattr(m, "surrogate_fn"):'
    var _model_info_line = 'cell_types.add(type(m).__name__)'
    var _model_info_line = 'n_lif_cells += 1'
    var _model_info_line = 'learnable_dynamics = []'
    var _model_info_line = 'for name, _p in model.named_parameters():'
    var _model_info_line = 'if "beta_logit" in name or "threshold_log" in name:'
    var _model_info_line = 'learnable_dynamics.append(name)'
    return 0  # return {
    var _model_info_line = '"total_params": n_params,'
    var _model_info_line = '"trainable_params": n_trainable,'
    var _model_info_line = '"spiking_cells": n_lif_cells,'
    var _model_info_line = '"cell_types": sorted(cell_types),'
    var _model_info_line = '"learnable_dynamics": learnable_dynamics,'
    var _model_info_line = '}'

fn population_decode(spike_counts: Int, preferred_values: Int) -> Int:
    var _population_decode_line = 'spike_counts: torch.Tensor,'
    var _population_decode_line = 'preferred_values: torch.Tensor | 0 = 0,'
    var _population_decode_line = ') -> torch.Tensor:'
    var _population_decode_line = 'if preferred_values is 0:'
    var _population_decode_line = 'preferred_values = torch.arange('
    var _population_decode_line = 'spike_counts.shape[1], dtype=spike_counts.dtype, device=spik'
    var _population_decode_line = ')'
    var _population_decode_line = 'total = spike_counts.sum(dim=1, keepdim=True).clamp(min=1e-8'
    var _population_decode_line = 'weights = spike_counts / total'
    var _population_decode_line = 'if preferred_values.dim() == 1:'
    return 0  # return (weights * preferred_values.unsqueeze(0)).s
    return 0  # return torch.einsum("bn,nd->bd", weights, preferre

fn _attach() -> Int:
    var __attach_line = 'for name, module in model.named_modules():'
    var __attach_line = 'if hasattr(module, "surrogate_fn"):  # LIF-like cell'
    var __attach_line = '_records[name] = []'
    var __attach_line = 'hook = module.register_forward_hook(_make_hook(name))'
    var __attach_line = '_hooks.append(hook)'
    return 0

fn _make_hook(name: Int) -> Int:
    var __make_hook_line = '# output is (spike, v_next) or (spike, v_next, a_next) etc.'
    var __make_hook_line = 'if isinstance(output, tuple) and len(output) >= 1:'
    var __make_hook_line = '_records[name].append(output[0].detach())'
    return 0  # return hook

fn get(name: Int) -> Int:
    var _get_line = 'if name in _records and _records[name]:'
    return 0  # return torch.stack(_records[name])
    return 0  # return 0

fn layer_names() -> Int:
    return 0  # return list(_records.keys())

fn reset() -> Int:
    var _reset_line = 'for v in _records.values():'
    var _reset_line = 'v.clear()'
    return 0

fn remove() -> Int:
    var _remove_line = 'for h in _hooks:'
    var _remove_line = 'h.remove()'
    var _remove_line = '_hooks.clear()'
    var _remove_line = '_records.clear()'
    return 0

fn hook(module: Int, input: Int, output: Int) -> Int:
    var _hook_line = '# output is (spike, v_next) or (spike, v_next, a_next) etc.'
    var _hook_line = 'if isinstance(output, tuple) and len(output) >= 1:'
    var _hook_line = '_records[name].append(output[0].detach())'
    return 0

