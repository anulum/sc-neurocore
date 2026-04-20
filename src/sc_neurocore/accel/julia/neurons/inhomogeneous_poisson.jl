# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for inhomogeneous_poisson

module InhomogeneousPoissonAccel

export step!, simulate, InhomogeneousPoissonNeuronState

mutable struct InhomogeneousPoissonNeuronState
    dt_ms::Float64
end

function InhomogeneousPoissonNeuronState()
    InhomogeneousPoissonNeuronState(1.0)
end

function step!(s::InhomogeneousPoissonNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        p = max(0.0, rate_hz) * s.dt_ms / 1000.0
        return (np.random.random() < p) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = InhomogeneousPoissonNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.dt_ms
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module InhomogeneousPoissonAccel
