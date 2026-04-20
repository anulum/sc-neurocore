# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for quantum_inspired_lif

module QuantumInspiredLifAccel

export step!, simulate, QuantumInspiredLIFNeuronState

mutable struct QuantumInspiredLIFNeuronState
    tau::Float64
    theta::Float64
    dt::Float64
    v_reset::Float64
    seed::Float64
    z_re::Float64
    z_im::Float64
    _rng_state::Float64
end

function QuantumInspiredLIFNeuronState()
    QuantumInspiredLIFNeuronState(20.0, 1.0, 0.1, 0.0, 12345.0, 0.0, 0.0, 0.0)
end

function _xorshift64(s::QuantumInspiredLIFNeuronState)
    x = s._rng_state & 18446744073709551615
    x = x << 13 & 18446744073709551615
    x = x >> 7 & 18446744073709551615
    x = x << 17 & 18446744073709551615
    s._rng_state = x
    return (x & 4294967295) / 4294967296.0
end

function step_complex(s::QuantumInspiredLIFNeuronState, i_re, i_im)
    dz_re = (-s.z_re + i_re) / s.tau
    dz_im = (-s.z_im + i_im) / s.tau
    s.z_re += dz_re * s.dt
    s.z_im += dz_im * s.dt
    prob = (s.z_re ^ 2 + s.z_im ^ 2) / s.theta ^ 2
    uniform = s._xorshift64()
    if uniform < min(prob, 1.0)
        s.z_re = s.v_reset
        s.z_im = s.v_reset
        return 1
    end
    return 0
end

function step!(s::QuantumInspiredLIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        return s.step_complex(I_ext, 0.0)
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = QuantumInspiredLIFNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.tau
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module QuantumInspiredLifAccel
