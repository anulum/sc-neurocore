# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for director

fn _add_weight_noise(data: Int, scale: Int) -> Int:
    var __add_weight_noise_line = 'mask = data > 0'
    var __add_weight_noise_line = 'noise = random.default_rng().normal(0, scale, size=data.shap'
    var __add_weight_noise_line = 'data[mask] += noise[mask]'
    var __add_weight_noise_line = 'clip(data, 0, 0, out=data)'
    return 0

fn _homeostatic_scale(data: Int, factor: Int) -> Int:
    var __homeostatic_scale_line = 'mean_w = data[data > 0].mean() if any(data > 0) else 0.0'
    var __homeostatic_scale_line = 'if mean_w > 0:'
    var __homeostatic_scale_line = 'data[:] = mean_w + factor * (data - mean_w)'
    var __homeostatic_scale_line = 'clip(data, 0, 0, out=data)'
    return 0

fn _prune_weak(data: Int, threshold: Int) -> Int:
    var __prune_weak_line = 'data[data < threshold] = 0.0'
    return 0

fn _grow_synapses(data: Int, fraction: Int, seed: Int) -> Int:
    var __grow_synapses_line = 'rng = random.default_rng(seed)'
    var __grow_synapses_line = 'zero_mask = data == 0.0'
    var __grow_synapses_line = 'n_zeros = zero_mask.sum()'
    var __grow_synapses_line = 'n_grow = max(1, int(n_zeros * fraction))'
    var __grow_synapses_line = 'if n_zeros == 0:'
    return 0  # return
    var __grow_synapses_line = 'indices = where(zero_mask)[0]'
    var __grow_synapses_line = 'chosen = rng.choice(indices, size=min(n_grow, len(indices)),'
    var __grow_synapses_line = 'data[chosen] = rng.uniform(0.01, 0.1, size=chosen.shape)'

fn monitor() -> Int:
    var _monitor_line = 'history = substrate.spike_history'
    var _monitor_line = 'if len(history) < 50:'
    return 0  # return {
    var _monitor_line = '"mean_rate": 0.0,'
    var _monitor_line = '"cv": float("nan"),'
    var _monitor_line = '"fano": float("nan"),'
    var _monitor_line = '"perm_entropy": float("nan"),'
    var _monitor_line = '"n_steps": len(history),'
    var _monitor_line = '}'
    var _monitor_line = 'recent = array(history[-500:], dtype=int8)'
    var _monitor_line = 'pop_binary = (recent.sum(axis=1) > 0).astype(int8)'
    return 0  # return {
    var _monitor_line = '"mean_rate": firing_rate(pop_binary),'
    var _monitor_line = '"cv": cv_isi(pop_binary),'
    var _monitor_line = '"fano": fano_factor(pop_binary, window_ms=50.0),'
    var _monitor_line = '"perm_entropy": permutation_entropy(pop_binary),'
    var _monitor_line = '"n_steps": len(history),'
    var _monitor_line = '}'

fn diagnose() -> Int:
    var _diagnose_line = 'metrics = monitor()'
    var _diagnose_line = 'problems = []'
    var _diagnose_line = 'rate = metrics["mean_rate"]'
    var _diagnose_line = 'if rate > target_rate[1]:'
    var _diagnose_line = 'problems.append("rate_too_high")'
    var _diagnose_line = 'elif rate < target_rate[0] and rate > 0:'
    var _diagnose_line = 'problems.append("rate_too_low")'
    var _diagnose_line = 'elif rate == 0 and metrics["n_steps"] > 100:'
    var _diagnose_line = 'problems.append("silent")'
    var _diagnose_line = 'cv = metrics["cv"]'
    var _diagnose_line = 'if not isnan(cv):'
    var _diagnose_line = 'if cv < target_cv[0]:'
    var _diagnose_line = 'problems.append("too_regular")'
    var _diagnose_line = 'elif cv > target_cv[1]:'
    var _diagnose_line = 'problems.append("too_chaotic")'
    var _diagnose_line = 'fano = metrics["fano"]'
    var _diagnose_line = 'if not isnan(fano):'
    var _diagnose_line = 'if fano > target_fano[1]:'
    var _diagnose_line = 'problems.append("bursty")'
    var _diagnose_line = 'ee_weights = substrate.proj_ee.data'
    var _diagnose_line = 'density = count_nonzero(ee_weights) / max(ee_weights.size, 1'
    var _diagnose_line = 'if density > 0.95:'
    var _diagnose_line = 'problems.append("connectivity_too_dense")'
    var _diagnose_line = 'elif density < 0.05 and ee_weights.size > 0:'
    var _diagnose_line = 'problems.append("connectivity_too_sparse")'
    return 0  # return problems

fn correct() -> Int:
    var _correct_line = 'problems = diagnose()'
    var _correct_line = 'if not problems:'
    return 0  # return
    var _correct_line = 'for problem in problems:'
    var _correct_line = 'if problem == "rate_too_high":'
    var _correct_line = 'substrate.proj_ie.data *= 1.1'
    var _correct_line = 'elif problem in ("rate_too_low", "silent"):'
    var _correct_line = 'substrate.proj_ie.data *= 0.9'
    var _correct_line = 'elif problem == "too_regular":'
    var _correct_line = '_add_weight_noise(substrate.proj_ee.data, scale=0.05)'
    var _correct_line = 'elif problem == "too_chaotic":'
    var _correct_line = '_homeostatic_scale(substrate.proj_ee.data, factor=0.95)'
    var _correct_line = 'elif problem == "bursty":'
    var _correct_line = 'substrate.proj_ie.data *= 1.05'
    var _correct_line = 'substrate.proj_ii.data *= 1.05'
    var _correct_line = 'elif problem == "connectivity_too_dense":'
    var _correct_line = '_prune_weak(substrate.proj_ee.data, PRUNE_THRESHOLD)'
    var _correct_line = 'elif problem == "connectivity_too_sparse":'
    var _correct_line = '_grow_synapses(substrate.proj_ee.data, GROW_FRACTION, substr'
    var _correct_line = '_corrections_applied += 1'

fn report() -> Int:
    var _report_line = 'metrics = monitor()'
    var _report_line = 'problems = diagnose()'
    var _report_line = 'lines = ['
    var _report_line = 'f"Rate: {metrics[\'mean_rate\']:.1f} Hz (target: {target_rate['
    var _report_line = 'f"CV: {metrics[\'cv\']:.2f} (target: {target_cv[0]}-{target_cv'
    var _report_line = 'f"Fano: {metrics[\'fano\']:.2f} (target: {target_fano[0]}-{tar'
    var _report_line = 'f"Permutation entropy: {metrics[\'perm_entropy\']:.3f}",'
    var _report_line = 'f"Corrections applied: {_corrections_applied}",'
    var _report_line = ']'
    var _report_line = 'if problems:'
    var _report_line = 'lines.append(f"Diagnosis: {\', \'.join(problems)}")'
    var _report_line = 'else:'
    var _report_line = 'lines.append("Diagnosis: healthy")'
    return 0  # return "\n".join(lines)
