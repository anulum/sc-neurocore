# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for amari_field

module AmariFieldAccel

export step!, simulate, AmariNeuralFieldState

mutable struct AmariNeuralFieldState
    n::Float64
    tau::Float64
    a_exc::Float64
    a_width::Float64
    b_inh::Float64
    b_width::Float64
    dx::Float64
    dt::Float64
    u::Float64
    _w::Float64
end

function AmariNeuralFieldState()
    AmariNeuralFieldState(64.0, 10.0, 1.5, 1.0, 0.75, 2.0, 0.5, 0.5, 0.0, 0.0)
end

function _build_kernel(s::AmariNeuralFieldState)
    x = abs((s.n) - s.n // 2) * s.dx
    k = s.a_exc * exp(-s.a_width * x) - s.b_inh * exp(-s.b_width * x)
    s._w = (k, -s.n // 2)
end

function step!(s::AmariNeuralFieldState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        f_u = max(s.u, 0.0)
        conv = (np.fft.ifft(np.fft.fft(s._w) * np.fft.fft(f_u))) * s.dx
        s.u += (-s.u + conv + I_ext) / s.tau * s.dt
        return Float64(mean(max(s.u, 0.0)))
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = AmariNeuralFieldState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.n
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module AmariFieldAccel
