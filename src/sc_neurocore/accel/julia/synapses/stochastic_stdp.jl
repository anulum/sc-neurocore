# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for synapses/stochastic_stdp

module StochasticStdpAccel

using Statistics, LinearAlgebra

mutable struct StochasticSTDPSynapseState
    learning_rate::Float64
    window_size::Float64
    ltd_ratio::Float64
    _pre_trace::Float64
end

function StochasticSTDPSynapseState()
    StochasticSTDPSynapseState(0.0, 0.0, 0.0, 0.0)
end

function process_step(s::StochasticSTDPSynapseState, pre_bit, post_bit)
    weight_bit = 1 if s._rng.random() < s.effective_weight_probability() else 0
    output_bit = pre_bit & weight_bit
    s._pre_trace = np.roll(s._pre_trace, 1)
    s._pre_trace[0] = pre_bit
    # Trace-based STDP: post spike + recent pre activity → LTP.
    # Pre spike without post → LTD. Mutually exclusive per timestep.
    if post_bit == 1 && np.any(s._pre_trace[1:])
        if s._rng.random() < s.learning_rate
            s._potentiate()
    elseif pre_bit == 1 && post_bit == 0
        if s._rng.random() < s.learning_rate * s.ltd_ratio
            s._depress()
    return output_bit
end

function _potentiate(s::StochasticSTDPSynapseState)
    new_w = min(s.w_max, s.w + s.learning_rate * (s.w_max - s.w_min))
    s.update_weight(new_w)
end

function _depress(s::StochasticSTDPSynapseState)
    new_w = max(s.w_min, s.w - s.learning_rate * (s.w_max - s.w_min))
    s.update_weight(new_w)
end

end # module StochasticStdpAccel
