# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for rall_cable

fn step(current: Int) -> Int:
    var _step_line = 'v_prev_soma = v[0]'
    var _step_line = 'dv = zeros(n_comp)'
    var _step_line = 'for i in range(n_comp):'
    var _step_line = 'leak = -(v[i] - v_rest)'
    var _step_line = 'left = v[i - 1] if i > 0 else v[i]'
    var _step_line = 'right = v[i + 1] if i < n_comp - 1 else v[i]'
    var _step_line = 'axial = g_ratio * (left - 2.0 * v[i] + right)'
    var _step_line = 'inj = current if i == n_comp - 1 else 0.0'
    var _step_line = 'dv[i] = (leak + axial + inj) / tau_m'
    var _step_line = 'v += dv * dt'
    var _step_line = 'if v[0] >= v_threshold and v_prev_soma < v_threshold:'
    var _step_line = 'v[0] = v_reset'
    return 0  # return 1
    return 0  # return 0

fn reset() -> Int:
    var _reset_line = 'v[:] = v_rest'
    return 0

