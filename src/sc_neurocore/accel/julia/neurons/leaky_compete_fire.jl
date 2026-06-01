# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for leaky_compete_fire

module LeakyCompeteFireAccel

export step!, simulate, LeakyCompeteFireNeuronState

mutable struct LeakyCompeteFireNeuronState
    n_units::Int64
    v::Vector{Float64}
    tau::Float64
    v_threshold::Float64
    w_inh::Float64
    dt::Float64
end

function LeakyCompeteFireNeuronState()
    LeakyCompeteFireNeuronState(4, zeros(4), 10.0, 1.0, 0.5, 1.0)
end

function lcf_finite(value::Float64)::Bool
    isfinite(value)
end

function validate_lcf(s::LeakyCompeteFireNeuronState, step_dt::Float64)::Nothing
    if s.n_units <= 0
        error("LCF n_units must be positive")
    end
    if length(s.v) != s.n_units
        error("LCF voltage vector length must match n_units")
    end
    if !(lcf_finite(s.tau) && s.tau > 0.0 && lcf_finite(step_dt) && step_dt > 0.0)
        error("LCF tau and dt must be finite and positive")
    end
    if !(lcf_finite(s.v_threshold) && lcf_finite(s.w_inh) && s.w_inh >= 0.0)
        error("LCF threshold and inhibition weight must be finite and valid")
    end
    if any(!isfinite(voltage) for voltage in s.v)
        error("LCF voltage vector must contain only finite values")
    end
    return nothing
end

function normalise_lcf_currents(currents::Vector{Float64}, n_units::Int64)::Vector{Float64}
    if length(currents) != n_units
        error("LCF currents must match n_units")
    end
    if any(!isfinite(current) for current in currents)
        error("LCF currents must contain only finite values")
    end
    return copy(currents)
end

function normalise_lcf_currents(current::Float64, n_units::Int64)::Vector{Float64}
    if !isfinite(current)
        error("LCF currents must contain only finite values")
    end
    return fill(current, n_units)
end

function step!(s::LeakyCompeteFireNeuronState, currents::Vector{Float64};
               dt::Float64=s.dt)::Vector{Int64}
    validate_lcf(s, dt)
    current_values = normalise_lcf_currents(currents, s.n_units)
    decay = exp(-dt / s.tau)
    next_v = [
        current_values[i] + (s.v[i] - current_values[i]) * decay
        for i in 1:s.n_units
    ]
    if any(!isfinite(voltage) for voltage in next_v)
        error("LCF exact relaxation produced a non-finite candidate")
    end
    spikes = zeros(Int64, s.n_units)
    for i in 1:s.n_units
        if next_v[i] >= s.v_threshold
            spikes[i] = 1
            next_v[i] = 0.0
            for j in 1:s.n_units
                if j != i
                    next_v[j] = max(0.0, next_v[j] - s.w_inh)
                end
            end
        end
    end
    s.v = next_v
    return spikes
end

function step!(s::LeakyCompeteFireNeuronState, current::Float64=0.0;
               dt::Float64=s.dt)::Vector{Int64}
    return step!(s, normalise_lcf_currents(current, s.n_units); dt=dt)
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = LeakyCompeteFireNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v[1]
        spikes += sum(result)
    end
    return trace, spikes
end

end # module LeakyCompeteFireAccel
