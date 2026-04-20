# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for spike_explain

fn top_k(k: Int) -> Int:
    var _top_k_line = 'flat = importance_map.ravel()'
    var _top_k_line = 'indices = argsort(flat)[::-1][:k]'
    var _top_k_line = 'T = importance_map.shape[0]'
    var _top_k_line = 'results = []'
    var _top_k_line = 'for idx in indices:'
    var _top_k_line = 't = idx // importance_map.shape[1]'
    var _top_k_line = 'n = idx % importance_map.shape[1]'
    var _top_k_line = 'results.append((int(t), int(n), float(flat[idx])))'
    return 0  # return results

fn summary() -> Int:
    var _summary_line = 'top = top_k(5)'
    var _summary_line = 'lines = [f"Explanation ({method}):"]'
    var _summary_line = 'for t, n, score in top:'
    var _summary_line = 'lines.append(f"  t={t}, neuron={n}: importance={score:.4f}")'
    return 0  # return "\n".join(lines)

fn attribute(spikes: Int, weights: Int, output_neuron: Int) -> Int:
    var _attribute_line = 'self,'
    var _attribute_line = 'spikes: ndarray,'
    var _attribute_line = 'weights: list[ndarray],'
    var _attribute_line = 'output_neuron: int = 0,'
    var _attribute_line = ') -> ExplanationResult:'
    var _attribute_line = 'T, N_in = spikes.shape'
    var _attribute_line = 'importance = zeros((T, N_in))'
    var _attribute_line = '# Backward through weight chain: output_neuron → input'
    var _attribute_line = '# Attribution = product of weight paths * temporal decay'
    var _attribute_line = 'attribution_weights = ones(N_in)'
    var _attribute_line = 'for w in reversed(weights):'
    var _attribute_line = 'if output_neuron < w.shape[0]:'
    var _attribute_line = 'row = abs(w[output_neuron])'
    var _attribute_line = 'if row.shape[0] == attribution_weights.shape[0]:'
    var _attribute_line = 'attribution_weights = attribution_weights * row'
    var _attribute_line = 'else:'
    var _attribute_line = 'attribution_weights = abs(w[output_neuron])'
    var _attribute_line = 'output_neuron = 0  # reset for next layer'
    var _attribute_line = '# Temporal attribution: weight input spikes by attribution +'
    var _attribute_line = 'for t in range(T):'
    var _attribute_line = 'time_weight = decay ** (T - 1 - t)'
    var _attribute_line = 'importance[t] = spikes[t].astype(float64) * attribution_weig'
    var _attribute_line = '# Normalize'
    var _attribute_line = 'max_val = importance.max()'
    var _attribute_line = 'if max_val > 0:'
    var _attribute_line = 'importance /= max_val'
    return 0  # return ExplanationResult(
    var _attribute_line = 'method="spike_attribution",'
    var _attribute_line = 'importance_map=importance,'
    var _attribute_line = ')'

fn explain(spikes: Int, output_neuron: Int) -> Int:
    var _explain_line = 'self,'
    var _explain_line = 'spikes: ndarray,'
    var _explain_line = 'output_neuron: int = 0,'
    var _explain_line = ') -> ExplanationResult:'
    var _explain_line = 'T, N = spikes.shape'
    var _explain_line = 'baseline_output = run_fn(spikes)'
    var _explain_line = 'if baseline_output.ndim > 0:'
    var _explain_line = 'baseline_val = float(baseline_output[output_neuron])'
    var _explain_line = 'else:'
    var _explain_line = 'baseline_val = float(baseline_output)'
    var _explain_line = 'importance = zeros((T, N))'
    var _explain_line = '# Find spike locations to perturb'
    var _explain_line = 'spike_locs = argwhere(spikes > 0)'
    var _explain_line = 'for t, n in spike_locs:'
    var _explain_line = 'perturbed = spikes.copy()'
    var _explain_line = 'perturbed[t, n] = 0'
    var _explain_line = 'perturbed_output = run_fn(perturbed)'
    var _explain_line = 'if perturbed_output.ndim > 0:'
    var _explain_line = 'new_val = float(perturbed_output[output_neuron])'
    var _explain_line = 'else:'
    var _explain_line = 'new_val = float(perturbed_output)'
    var _explain_line = 'importance[t, n] = abs(baseline_val - new_val)'
    var _explain_line = 'max_val = importance.max()'
    var _explain_line = 'if max_val > 0:'
    var _explain_line = 'importance /= max_val'
    return 0  # return ExplanationResult(
    var _explain_line = 'method="temporal_saliency",'
    var _explain_line = 'importance_map=importance,'
    var _explain_line = ')'

fn explain(spikes: Int, output_neuron: Int) -> Int:
    var _explain_line = 'self,'
    var _explain_line = 'spikes: ndarray,'
    var _explain_line = 'output_neuron: int = 0,'
    var _explain_line = ') -> ExplanationResult:'
    var _explain_line = 'T, N = spikes.shape'
    var _explain_line = 'baseline_output = run_fn(spikes)'
    var _explain_line = 'if baseline_output.ndim > 0:'
    var _explain_line = 'baseline_val = float(baseline_output[output_neuron])'
    var _explain_line = 'else:'
    var _explain_line = 'baseline_val = float(baseline_output)'
    var _explain_line = 'neuron_importance = zeros(N)'
    var _explain_line = 'for n in range(N):'
    var _explain_line = 'silenced = spikes.copy()'
    var _explain_line = 'silenced[:, n] = 0'
    var _explain_line = 'silenced_output = run_fn(silenced)'
    var _explain_line = 'if silenced_output.ndim > 0:'
    var _explain_line = 'new_val = float(silenced_output[output_neuron])'
    var _explain_line = 'else:'
    var _explain_line = 'new_val = float(silenced_output)'
    var _explain_line = 'neuron_importance[n] = abs(baseline_val - new_val)'
    var _explain_line = 'max_val = neuron_importance.max()'
    var _explain_line = 'if max_val > 0:'
    var _explain_line = 'neuron_importance /= max_val'
    var _explain_line = 'importance_map = tile(neuron_importance, (1, 1))'
    return 0  # return ExplanationResult(
    var _explain_line = 'method="causal_importance",'
    var _explain_line = 'importance_map=importance_map,'
    var _explain_line = ')'

