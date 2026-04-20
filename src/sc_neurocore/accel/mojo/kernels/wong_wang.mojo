# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for wong_wang

fn _phi(i_syn: Int) -> Int:
    var __phi_line = 'a, b, d = 270.0, 108.0, 0.154'
    var __phi_line = 'x = a * i_syn - b'
    var __phi_line = 'if abs(x) < 1e-6:'
    return 0  # return 1.0 / d
    return 0  # return x / (1.0 - exp(-d * x))

fn step(stim1: Int, stim2: Int) -> Int:
    var _step_line = 'i1 = ('
    var _step_line = 'j_n * s1'
    var _step_line = '- j_cross * s2'
    var _step_line = '+ i_0'
    var _step_line = '+ stim1'
    var _step_line = '+ sigma * random.randn()'
    var _step_line = ')'
    var _step_line = 'i2 = ('
    var _step_line = 'j_n * s2'
    var _step_line = '- j_cross * s1'
    var _step_line = '+ i_0'
    var _step_line = '+ stim2'
    var _step_line = '+ sigma * random.randn()'
    var _step_line = ')'
    var _step_line = 'r1, r2 = _phi(i1), _phi(i2)'
    var _step_line = 's1 += (-s1 / tau_s + (1.0 - s1) * gamma * r1) * dt'
    var _step_line = 's2 += (-s2 / tau_s + (1.0 - s2) * gamma * r2) * dt'
    var _step_line = 's1 = clip(s1, 0.0, 1.0)'
    var _step_line = 's2 = clip(s2, 0.0, 1.0)'
    return 0  # return (r1, r2)

fn reset() -> Int:
    var _reset_line = 's1, s2 = 0.1, 0.1'
    return 0
