# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for population

fn _resolve_model(model: Int) -> Int:
    var __resolve_model_line = 'if isinstance(model, str):'
    var __resolve_model_line = 'cls = getattr(_model_registry, model, 0)'
    var __resolve_model_line = 'if cls is 0:'
    var __resolve_model_line = 'raise ValueError(f"Unknown model \'{model}\'. Check neurons.mo'
    return 0  # return cls
    return 0  # return model

fn _sync_voltages() -> Int:
    var __sync_voltages_line = 'for i, neuron in enumerate(neurons):'
    var __sync_voltages_line = '_voltages[i] = getattr(neuron, "v", 0.0)'
    return 0

fn step_all(currents: Int, spike_gating: Int) -> Int:
    var _step_all_line = 'spikes = zeros(n, dtype=int8)'
    var _step_all_line = 'if spike_gating:'
    var _step_all_line = 'for i, neuron in enumerate(neurons):'
    var _step_all_line = 'v = getattr(neuron, "v", 0.0)'
    var _step_all_line = 'v_thresh = getattr(neuron, "v_threshold", 1.0)'
    var _step_all_line = 'v_rest = getattr(neuron, "v_rest", 0.0)'
    var _step_all_line = '# Skip if no input AND voltage within 1% of rest'
    var _step_all_line = 'if currents[i] == 0.0 and abs(v - v_rest) < 0.01 * abs(v_thr'
    var _step_all_line = 'continue'
    var _step_all_line = 'raw = neuron.step(float(currents[i]))'
    var _step_all_line = 'spikes[i] = min(max(int(raw), 0), 1)'
    var _step_all_line = '_voltages[i] = getattr(neuron, "v", 0.0)'
    var _step_all_line = 'else:'
    var _step_all_line = 'for i, neuron in enumerate(neurons):'
    var _step_all_line = 'raw = neuron.step(float(currents[i]))'
    var _step_all_line = 'spikes[i] = min(max(int(raw), 0), 1)'
    var _step_all_line = '_voltages[i] = getattr(neuron, "v", 0.0)'
    return 0  # return spikes

fn reset_all() -> Int:
    var _reset_all_line = 'for neuron in neurons:'
    var _reset_all_line = 'if hasattr(neuron, "reset"):'
    var _reset_all_line = 'neuron.reset()'
    var _reset_all_line = 'elif hasattr(neuron, "reset_state"):'
    var _reset_all_line = 'neuron.reset_state()'
    var _reset_all_line = '_sync_voltages()'
    return 0

fn get_states() -> Int:
    var _get_states_line = 'if n == 0:'
    return 0  # return {}
    var _get_states_line = 'sample = neurons[0]'
    var _get_states_line = 'if hasattr(sample, "get_state"):'
    var _get_states_line = 'keys = sample.get_state().keys()'
    var _get_states_line = 'elif hasattr(sample, "__dataclass_fields__"):'
    var _get_states_line = 'keys = [k for k in sample.__dataclass_fields__ if k not in ('
    var _get_states_line = 'else:'
    var _get_states_line = 'keys = ["v"]'
    var _get_states_line = 'result = {}'
    var _get_states_line = 'for k in keys:'
    var _get_states_line = 'result[k] = array([getattr(n, k, 0.0) for n in neurons])'
    return 0  # return result

fn set_voltages(voltages: Int) -> Int:
    var _set_voltages_line = 'for i, neuron in enumerate(neurons):'
    var _set_voltages_line = 'if hasattr(neuron, "v"):'
    var _set_voltages_line = 'neuron.v = float(voltages[i])'
    var _set_voltages_line = '_voltages[:] = voltages[: n]'
    return 0

fn voltages() -> Int:
    return 0  # return _voltages

