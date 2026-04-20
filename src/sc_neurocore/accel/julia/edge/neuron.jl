# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for edge/neuron

module NeuronAccel

using Statistics, LinearAlgebra

mutable struct IzhikevichNeuronState
    threshold::Float64
    leak_shift::Float64
    membrane::Float64
    spike_count::Float64
    a_q16::Float64
    b_q16::Float64
    c_q16::Float64
    d_q16::Float64
    v_q16::Float64
    u_q16::Float64
    _q16_one::Float64
end

function IzhikevichNeuronState()
    IzhikevichNeuronState(512.0, 3.0, 0.0, 0.0, 1311.0, 13107.0, -4259840.0, 524288.0, -4259840.0, -917504.0, 0.0)
end

function tick(s::IzhikevichNeuronState, input_words)
    excitation = popcount_slice(input_words)
    s.membrane += excitation
    s.membrane -= (s.membrane >> s.leak_shift)
    if s.membrane >= s.threshold
        s.membrane = 0
        s.spike_count += 1
        return true
    return false
end

function reset(s::IzhikevichNeuronState)
    s.membrane = 0
    s.spike_count = 0
end

function tick(s::IzhikevichNeuronState, input_current_q16)
    v = s.v_q16
    u = s.u_q16
    dv = ((v * v) >> 14) + ((5 * v) >> 0) + (140 << 16) - u + input_current_q16
    du = (s.a_q16 * ((s.b_q16 * v >> 16) - u)) >> 16
    s.v_q16 = v + (dv >> 8)
    s.u_q16 = u + (du >> 8)
    if s.v_q16 >= (30 << 16)
        s.v_q16 = s.c_q16
        s.u_q16 += s.d_q16
        s.spike_count += 1
        return true
    return false
end

function reset(s::IzhikevichNeuronState)
    s.v_q16 = s.c_q16
    s.u_q16 = -917504
    s.spike_count = 0
end

function regular_spiking(s::IzhikevichNeuronState)
    return cls(a_q16=1311, b_q16=13107, c_q16=-4259840, d_q16=524288)
end

function fast_spiking(s::IzhikevichNeuronState)
    return cls(a_q16=6554, b_q16=13107, c_q16=-4259840, d_q16=131072)
end

function chattering(s::IzhikevichNeuronState)
    return cls(a_q16=1311, b_q16=13107, c_q16=-3276800, d_q16=131072)
end

function intrinsic_burst(s::IzhikevichNeuronState)
    return cls(a_q16=1311, b_q16=13107, c_q16=-3604480, d_q16=262144)
end

end # module NeuronAccel
