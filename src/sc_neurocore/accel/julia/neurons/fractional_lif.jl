# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for fractional_lif

module FractionalLifAccel

export step!, simulate, FractionalLIFNeuronState

mutable struct FractionalLIFNeuronState
    v::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    alpha::Float64
    resistance::Float64
    dt::Float64
    _max_history::Float64
end

function FractionalLIFNeuronState()
    FractionalLIFNeuronState(0.0, 0.0, 0.0, 1.0, 0.8, 1.0, 1.0, 100.0)
end

function _compute_gl_coefficients(s::FractionalLIFNeuronState)
    coeffs = [1.0]
    for k in 1:s._max_history
        coeffs.append(coeffs[-1] * (k - 1 - s.alpha) / k)
    end
    return coeffs
end

function step!(s::FractionalLIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        rhs = -(s.v - s.v_rest) + s.resistance * I_ext
        history = s._history
        gl_sum = sum((s._gl_coeffs[k] * history[-(k + 1)] for k in range(1, min(length(history), s._max_history)) if length(history) > k))
        s.v = rhs * s.dt ^ s.alpha - gl_sum
        history.append(s.v)
        if length(history) > s._max_history
            history.pop(0)
        end
        if s.v >= s.v_threshold
            s.v = s.v_reset
            history[-1] = s.v_reset
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = FractionalLIFNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module FractionalLifAccel
