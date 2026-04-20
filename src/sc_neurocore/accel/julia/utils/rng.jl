# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for utils/rng

module RngAccel

using Statistics, LinearAlgebra

mutable struct RNGState
    _rng::Float64
end

function RNGState()
    RNGState(0.0)
end

function normal(s::RNGState)
    self, mean: float = 0.0, std: float = 1.0, size: int | tuple[int, ...] | nothing = nothing
    ) -> Any
    return s._rng.normal(mean, std, size)
end

function uniform(s::RNGState)
    self, low: float = 0.0, high: float = 1.0, size: int | tuple[int, ...] | nothing = nothing
    ) -> Any
    return s._rng.uniform(low, high, size)
end

function bernoulli(s::RNGState, p, size, ...] | None)
    return s._rng.random(size) < p
end

function random(s::RNGState, size, ...] | None)
    return s._rng.random(size)
end

function shuffle(s::RNGState, x)
    s._rng.shuffle(x)
end

end # module RngAccel
