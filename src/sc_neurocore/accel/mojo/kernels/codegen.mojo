# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for codegen

fn generate_model_script(model_name: Int, params: Int, duration: Int, current: Int, dt: Int) -> Int:
    var _generate_model_script_line = 'model_name: str,'
    var _generate_model_script_line = 'params: dict[str, float] | 0 = 0,'
    var _generate_model_script_line = 'duration: float = 100.0,'
    var _generate_model_script_line = 'current: float = 10.0,'
    var _generate_model_script_line = 'dt: float = 0.1,'
    var _generate_model_script_line = ') -> str:'
    var _generate_model_script_line = 'param_args = ""'
    var _generate_model_script_line = 'if params:'
    var _generate_model_script_line = 'non_default = {k: v for k, v in params.items()}'
    var _generate_model_script_line = 'if non_default:'
    var _generate_model_script_line = 'param_args = ", ".join(f"{k}={v}" for k, v in non_default.it'
    var _generate_model_script_line = 'n_steps = int(duration / dt)'
    return 0

fn generate_ode_script(equations: Int, threshold: Int, reset: Int, params: Int, init: Int, duration: Int) -> Int:
    var _generate_ode_script_line = 'equations: list[str],'
    var _generate_ode_script_line = 'threshold: str | 0 = 0,'
    var _generate_ode_script_line = 'reset: str | 0 = 0,'
    var _generate_ode_script_line = 'params: dict[str, float] | 0 = 0,'
    var _generate_ode_script_line = 'init: dict[str, float] | 0 = 0,'
    var _generate_ode_script_line = 'duration: float = 100.0,'
    var _generate_ode_script_line = 'current: float = 10.0,'
    var _generate_ode_script_line = 'dt: float = 0.1,'
    var _generate_ode_script_line = ') -> str:'
    var _generate_ode_script_line = 'eq_lines = ",\\n        ".join(f\'"{eq}"\' for eq in equations)'
    var _generate_ode_script_line = 'param_str = repr(params) if params else "{}"'
    var _generate_ode_script_line = 'init_str = repr(init) if init else "{}"'
    var _generate_ode_script_line = 'n_steps = int(duration / dt)'
    return 0

fn generate_oneliner(model_name: Int, params: Int, current: Int) -> Int:
    var _generate_oneliner_line = 'model_name: str | 0 = 0,'
    var _generate_oneliner_line = 'params: dict[str, float] | 0 = 0,'
    var _generate_oneliner_line = 'current: float = 10.0,'
    var _generate_oneliner_line = ') -> str:'
    var _generate_oneliner_line = 'if model_name:'
    var _generate_oneliner_line = 'args = ", ".join(f"{k}={v}" for k, v in (params or {}).items'
    return 0  # return f"from sc_neurocore.neurons.models import {
    return 0  # return ""

fn classify_firing_pattern(spikes: Int, n_steps: Int, dt: Int) -> Int:
    var _classify_firing_pattern_line = 'spikes: list[int],'
    var _classify_firing_pattern_line = 'n_steps: int,'
    var _classify_firing_pattern_line = 'dt: float,'
    var _classify_firing_pattern_line = ') -> dict:'
    var _classify_firing_pattern_line = 'if len(spikes) == 0:'
    return 0  # return {"pattern": "silent", "description": "No sp
    var _classify_firing_pattern_line = 'duration_s = n_steps * dt / 1000.0'
    var _classify_firing_pattern_line = 'rate = len(spikes) / duration_s if duration_s > 0 else 0'
    var _classify_firing_pattern_line = 'if len(spikes) < 3:'
    return 0  # return {
    var _classify_firing_pattern_line = '"pattern": "single_spike",'
    var _classify_firing_pattern_line = '"description": f"Only {len(spikes)} spike(s)",'
    var _classify_firing_pattern_line = '"rate_hz": round(rate, 1),'
    var _classify_firing_pattern_line = '}'
    var _classify_firing_pattern_line = 'import numpy as np'
    var _classify_firing_pattern_line = 'isis = diff(spikes).astype(float) * dt'
    var _classify_firing_pattern_line = 'isi_mean = float(mean(isis))'
    var _classify_firing_pattern_line = 'isi_cv = float(std(isis) / isi_mean) if isi_mean > 0 else 0'
    var _classify_firing_pattern_line = '# Detect bursting: look for bimodal ISI (short intra-burst +'
    var _classify_firing_pattern_line = 'if len(isis) >= 4:'
    var _classify_firing_pattern_line = 'sorted_isis = sort(isis)'
    var _classify_firing_pattern_line = 'median_isi = float(median(isis))'
    var _classify_firing_pattern_line = 'short = isis[isis < median_isi * 0.5]'
    var _classify_firing_pattern_line = 'long = isis[isis > median_isi * 1.5]'
    var _classify_firing_pattern_line = 'if len(short) > 1 and len(long) > 0:'
    var _classify_firing_pattern_line = 'ratio = float(mean(long)) / float(mean(short)) if mean(short'
    var _classify_firing_pattern_line = 'if ratio > 3:'
    return 0  # return {
    var _classify_firing_pattern_line = '"pattern": "bursting",'
    var _classify_firing_pattern_line = '"description": f"Burst-pause pattern (ISI ratio {ratio:.1f}x'
    var _classify_firing_pattern_line = '"rate_hz": round(rate, 1),'
    var _classify_firing_pattern_line = '"isi_cv": round(isi_cv, 3),'
    var _classify_firing_pattern_line = '"burst_isi_ms": round(float(mean(short)), 2),'
    var _classify_firing_pattern_line = '"inter_burst_ms": round(float(mean(long)), 2),'
    var _classify_firing_pattern_line = '}'
    var _classify_firing_pattern_line = '# Detect adaptation: ISIs increase over time'
    var _classify_firing_pattern_line = 'if len(isis) >= 5:'
    var _classify_firing_pattern_line = 'first_third = mean(isis[: len(isis) // 3])'
    var _classify_firing_pattern_line = 'last_third = mean(isis[-len(isis) // 3 :])'
    var _classify_firing_pattern_line = 'if last_third > first_third * 1.3:'
    return 0  # return {
    var _classify_firing_pattern_line = '"pattern": "adapting",'
    var _classify_firing_pattern_line = '"description": f"Spike-frequency adaptation ({first_third:.1'
    var _classify_firing_pattern_line = '"rate_hz": round(rate, 1),'
    var _classify_firing_pattern_line = '"isi_cv": round(isi_cv, 3),'
    var _classify_firing_pattern_line = '}'
    var _classify_firing_pattern_line = 'if isi_cv < 0.15:'
    return 0  # return {
    var _classify_firing_pattern_line = '"pattern": "tonic",'
    var _classify_firing_pattern_line = '"description": f"Regular tonic firing (CV={isi_cv:.3f})",'
    var _classify_firing_pattern_line = '"rate_hz": round(rate, 1),'
    var _classify_firing_pattern_line = '"isi_cv": round(isi_cv, 3),'
    var _classify_firing_pattern_line = '}'
    var _classify_firing_pattern_line = 'if isi_cv < 0.5:'
    return 0  # return {
    var _classify_firing_pattern_line = '"pattern": "irregular",'
    var _classify_firing_pattern_line = '"description": f"Irregular spiking (CV={isi_cv:.3f})",'
    var _classify_firing_pattern_line = '"rate_hz": round(rate, 1),'
    var _classify_firing_pattern_line = '"isi_cv": round(isi_cv, 3),'
    var _classify_firing_pattern_line = '}'
    return 0  # return {
    var _classify_firing_pattern_line = '"pattern": "chaotic",'
    var _classify_firing_pattern_line = '"description": f"Highly irregular/chaotic (CV={isi_cv:.3f})"'
    var _classify_firing_pattern_line = '"rate_hz": round(rate, 1),'
    var _classify_firing_pattern_line = '"isi_cv": round(isi_cv, 3),'
    var _classify_firing_pattern_line = '}'
