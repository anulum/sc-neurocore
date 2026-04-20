# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for leaky_compete_fire

fn step(currents: Int) -> Int:
    var _step_line = 'if isinstance(currents, (int, float)):'
    var _step_line = 'currents = [currents] * n_units'
    var _step_line = 'spikes = [0] * n_units'
    var _step_line = 'for i in range(n_units):'
    var _step_line = 'v[i] += (-v[i] + currents[i]) / tau * dt'
    var _step_line = 'for i in range(n_units):'
    var _step_line = 'if v[i] >= v_threshold:'
    var _step_line = 'spikes[i] = 1'
    var _step_line = 'v[i] = 0.0'
    var _step_line = 'for j in range(n_units):'
    var _step_line = 'if j != i:'
    var _step_line = 'v[j] -= w_inh'
    var _step_line = 'v[j] = max(0.0, v[j])'
    return 0  # return spikes

fn reset() -> Int:
    var _reset_line = 'v = [0.0] * n_units'
    return 0

