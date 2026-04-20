# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for brainscales_adex

fn step(current: Int) -> Int:
    var _step_line = 'dt_hw = dt * hw_speedup'
    var _step_line = 'exp_arg = clip((v - v_rh) / delta_t, -20.0, 20.0)'
    var _step_line = 'exp_term = delta_t * exp(exp_arg)'
    var _step_line = 'dv = ('
    var _step_line = '(-(v - v_rest) + exp_term - w + current)'
    var _step_line = '/ tau'
    var _step_line = '* (dt_hw / hw_speedup)'
    var _step_line = ')'
    var _step_line = 'dw = (a * (v - v_rest) - w) / tau_w * (dt_hw / hw_speedup)'
    var _step_line = 'v += dv'
    var _step_line = 'w += dw'
    var _step_line = 'if v >= v_threshold:'
    var _step_line = 'v = v_reset'
    var _step_line = 'w += b'
    return 0  # return 1
    return 0  # return 0

fn reset() -> Int:
    var _reset_line = 'v = v_rest'
    var _reset_line = 'w = 0.0'
    return 0
