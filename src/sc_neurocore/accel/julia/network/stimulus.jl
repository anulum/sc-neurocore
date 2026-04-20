# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for network/stimulus

module StimulusAccel

using Statistics, LinearAlgebra

mutable struct StepCurrentState
    values::Float64
    dt::Float64
    n::Float64
    rate_hz::Float64
    weight::Float64
    _rng::Float64
    onset::Float64
    offset::Float64
    amplitude::Float64
end

function StepCurrentState()
    StepCurrentState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function get_current(s::StepCurrentState, t_step)
    idx = min(t_step, length(s.values) - 1)
    return float(s.values[idx])
end

function get_current(s::StepCurrentState, t_step, dt)
    step_dt = dt if dt is ! nothing else s.dt
    p_spike = s.rate_hz * step_dt
    spikes = (s._rng.random(s.n) < p_spike).astype(np.float64)
    return spikes * s.weight
end

function get_current(s::StepCurrentState, t_step, dt)
    if s.onset <= t_step < s.offset
        return s.amplitude
    return 0.0
end

end # module StimulusAccel
