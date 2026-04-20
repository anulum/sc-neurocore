# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for glif

fn step(current: Int) -> Int:
    var _step_line = 'dv = ('
    var _step_line = '(-(v - v_rest) + resistance * current + i_asc1 + i_asc2)'
    var _step_line = '/ tau_m'
    var _step_line = '* dt'
    var _step_line = ')'
    var _step_line = 'dtheta = ('
    var _step_line = '(theta_inf - theta + a_theta * (v - v_rest))'
    var _step_line = '/ tau_theta'
    var _step_line = '* dt'
    var _step_line = ')'
    var _step_line = 'i_asc1 *= exp(-dt / tau_asc1)'
    var _step_line = 'i_asc2 *= exp(-dt / tau_asc2)'
    var _step_line = 'v += dv'
    var _step_line = 'theta += dtheta'
    var _step_line = 'if v >= theta:'
    var _step_line = 'v = v_reset'
    var _step_line = 'theta += delta_theta'
    var _step_line = 'i_asc1 += r_asc1'
    var _step_line = 'i_asc2 += r_asc2'
    return 0  # return 1
    return 0  # return 0

fn reset() -> Int:
    var _reset_line = 'v = v_rest'
    var _reset_line = 'theta = theta_inf'
    var _reset_line = 'i_asc1, i_asc2 = 0.0, 0.0'
    return 0

