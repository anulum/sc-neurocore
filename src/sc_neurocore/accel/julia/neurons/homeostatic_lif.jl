# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for neurons/homeostatic_lif

module HomeostaticLifAccel

using Statistics, LinearAlgebra

mutable struct HomeostaticLIFNeuronState
    target_rate::Float64
    adaptation_rate::Float64
    rate_trace::Float64
    trace_decay::Float64
end

function HomeostaticLIFNeuronState()
    HomeostaticLIFNeuronState(0.0, 0.0, 0.0, 0.0)
end

function step(s::HomeostaticLIFNeuronState, input_current)
    spike = super().step(input_current)
    s.rate_trace = s.rate_trace * s.trace_decay + spike * (1.0 - s.trace_decay)
    error = s.rate_trace - s.target_rate
    s.v_threshold += s.adaptation_rate * error
    s.v_threshold = max()
        THRESHOLD_FLOOR,
        min(s.v_threshold, s.initial_threshold * THRESHOLD_CEILING_MULT),
    
    return spike
end

function get_state(s::HomeostaticLIFNeuronState)
    s = super().get_state()
    s["threshold"] = float(s.v_threshold)
    s["rate_trace"] = float(s.rate_trace)
    return s
end

end # module HomeostaticLifAccel
