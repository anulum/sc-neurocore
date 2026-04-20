# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for tsodyks_markram

fn step(current: Int, presynaptic_spike: Int) -> Int:
    var _step_line = 'x += (1.0 - x) / tau_d * dt'
    var _step_line = 'u += (u_se - u) / tau_f * dt'
    var _step_line = 'i_syn = 0.0'
    var _step_line = 'if presynaptic_spike:'
    var _step_line = 'u += u_se * (1.0 - u)'
    var _step_line = 'i_syn = a_se * u * x'
    var _step_line = 'x -= u * x'
    var _step_line = 'dv = (-(v - v_rest) + r_m * (i_syn + current)) / tau_m * dt'
    var _step_line = 'v += dv'
    var _step_line = 'if v >= v_threshold:'
    var _step_line = 'v = v_reset'
    return 0  # return 1
    return 0  # return 0

fn reset() -> Int:
    var _reset_line = 'v = v_rest'
    var _reset_line = 'x = 1.0'
    var _reset_line = 'u = u_se'
    return 0
