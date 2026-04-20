# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for robotics/cpg

module CpgAccel

using Statistics, LinearAlgebra

mutable struct StochasticCPGState
    drive_current::Float64
    inhibition_weight::Float64
end

function StochasticCPGState()
    StochasticCPGState(2.0, 2.0)
end

function step(s::StochasticCPGState)
    # Inhibition logic
    # Input to N1 = Drive - Weight * N2_Activity
    # Input to N2 = Drive - Weight * N1_Activity
    # We use a trace of spikes for inhibition "potential"
    i1 = s.drive_current - s.inhibition_weight * s.s2_trace
    i2 = s.drive_current - s.inhibition_weight * s.s1_trace
    spike1 = s.n1.step(i1)
    spike2 = s.n2.step(i2)
    # Update traces
    s.s1_trace = s.s1_trace * s.decay + spike1
    s.s2_trace = s.s2_trace * s.decay + spike2
    return spike1, spike2
end

end # module CpgAccel
