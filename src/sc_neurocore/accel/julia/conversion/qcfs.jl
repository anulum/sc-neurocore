# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for conversion/qcfs

module QcfsAccel

using Statistics, LinearAlgebra

mutable struct QCFSActivationState
    T::Float64
    theta::Float64
end

function QCFSActivationState()
    QCFSActivationState(0.0, 0.0)
end

function forward(s::QCFSActivationState, x)
    scaled = x * s.T / s.theta + 0.5
    # STE: floor in forward, pass gradient straight through
    quantized = scaled.floor() - (scaled.floor() - scaled).detach()
    clipped = quantized.clamp(0, s.T)
    return clipped * s.theta / s.T
end

function extra_repr(s::QCFSActivationState)
    return f"T={s.T}, theta={s.theta.item():.2f}"
end

end # module QcfsAccel
