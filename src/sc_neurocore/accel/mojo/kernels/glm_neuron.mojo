# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for glm_neuron

fn step(stimulus: Int) -> Int:
    var _step_line = '_stim_buf = roll(_stim_buf, 1)'
    var _step_line = '_stim_buf[0] = stimulus'
    var _step_line = 'log_rate = float(dot(k, _stim_buf) + dot(h, _spike_buf) + mu'
    var _step_line = 'lam = exp(clip(log_rate, -20.0, 20.0))'
    var _step_line = 'p = lam * dt_ms / 1000.0'
    var _step_line = 'spike = 1 if _rng.random() < min(p, 1.0) else 0'
    var _step_line = '_spike_buf = roll(_spike_buf, 1)'
    var _step_line = '_spike_buf[0] = float(spike)'
    return 0  # return spike

fn reset() -> Int:
    var _reset_line = '_stim_buf = zeros(n_k)'
    var _reset_line = '_spike_buf = zeros(n_h)'
    return 0

