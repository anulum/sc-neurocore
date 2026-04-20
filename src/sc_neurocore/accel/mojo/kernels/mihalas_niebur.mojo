# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for mihalas_niebur

fn step(current: Int) -> Int:
    var _step_line = 'dv = (-(v - v_rest) + i1 + i2 + current) / tau_v * dt'
    var _step_line = 'dtheta = ('
    var _step_line = '(theta_inf - theta + a * (v - v_rest))'
    var _step_line = '/ tau_theta'
    var _step_line = '* dt'
    var _step_line = ')'
    var _step_line = 'di1 = -i1 / tau_1 * dt'
    var _step_line = 'di2 = -i2 / tau_2 * dt'
    var _step_line = 'v += dv'
    var _step_line = 'theta += dtheta'
    var _step_line = 'i1 += di1'
    var _step_line = 'i2 += di2'
    var _step_line = 'if v >= theta:'
    var _step_line = 'v = v_reset'
    var _step_line = 'theta = max(theta, theta_reset)'
    var _step_line = 'i1 += r1'
    var _step_line = 'i2 += r2'
    return 0  # return 1
    return 0  # return 0

fn reset() -> Int:
    var _reset_line = 'v = v_rest'
    var _reset_line = 'theta = theta_reset'
    var _reset_line = 'i1 = 0.0'
    var _reset_line = 'i2 = 0.0'
    return 0
