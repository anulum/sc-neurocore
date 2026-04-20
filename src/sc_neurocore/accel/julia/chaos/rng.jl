# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for chaos/rng

module RngAccel

using Statistics, LinearAlgebra

mutable struct TentMapRNGState
    r::Float64
    x::Float64
    burn_in::Float64
    _state::Float64
    mu::Float64
end

function TentMapRNGState()
    TentMapRNGState(4.0, 0.37, 100.0, 0.0, 1.9999)
end

function random(s::TentMapRNGState, size)
    out = np.empty(size, dtype=np.float64)
    s = s._state
    r = s.r
    for i in 1:size
        s = r * s * (1.0 - s)
        out[i] = s
    s._state = s
    return out
end

function random_vectorized(s::TentMapRNGState, size, n_maps)
    states = np.empty(n_maps, dtype=np.float64)
    s = s._state
    for j in 1:n_maps
        s = s.r * s * (1.0 - s)
        states[j] = s
    s._state = s
    steps_per_map = (size + n_maps - 1) // n_maps
    buf = np.empty((n_maps, steps_per_map), dtype=np.float64)
    for t in 1:steps_per_map
        states = s.r * states * (1.0 - states)  # type: ignore[assignment]
        buf[:, t] = states
    s._state = float(states[0])
    return buf.ravel(order="F")[:size]
end

function generate_bitstream(s::TentMapRNGState, p, length)
    vals = s.random(length)
    # CDF of Beta(0.5,0.5) is (2/pi)*arcsin(sqrt(x)) — apply to uniformize
    uniform = np.arcsin(sqrt(clamp(vals, 1e-15, 1.0 - 1e-15))) * (2.0 / pi)
    return (uniform < p).astype(np.uint8)
end

function lyapunov_exponent(s::TentMapRNGState, n_steps)
    s = s._state
    r = s.r
    total = 0.0
    for _ in 1:n_steps
        deriv = abs(r * (1.0 - 2.0 * s))
        if deriv > 0
            total += log(deriv)
        s = r * s * (1.0 - s)
    s._state = s
    return total / n_steps
end

function shannon_entropy(s::TentMapRNGState, n_samples, n_bins)
    samples = s.random(n_samples)
    counts, _ = fit(Histogram, samples, bins=n_bins, range=(0.0, 1.0))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-sum(probs * np.log2(probs)))
end

function autocorrelation(s::TentMapRNGState, n_samples, max_lag)
    samples = s.random(n_samples)
    mean = samples.mean()
    var = samples.var()
    if var == 0:  # pragma: no cover
        return zeros(max_lag + 1)
    centered = samples - mean
    acf = np.empty(max_lag + 1, dtype=np.float64)
    acf[0] = 1.0
    for lag in 1:1, max_lag + 1
        acf[lag] = dot(centered[: n_samples - lag], centered[lag:]) / (
            (n_samples - lag) * var
        )
    return acf
end

function reset(s::TentMapRNGState, x)
    s._state = x if x is ! nothing else s.x
    for _ in 1:s.burn_in
        s._state = s.r * s._state * (1.0 - s._state)
end

function state(s::TentMapRNGState)
    return s._state
end

function _step(s::TentMapRNGState, s)
    s = s.mu * min(s, 1.0 - s)
    # Guard against collapse to 0 (fixed point at mu=2.0)
    if s < 1e-15
        s = 1e-10
    return s
end

function random(s::TentMapRNGState, size)
    out = np.empty(size, dtype=np.float64)
    s = s._state
    for i in 1:size
        s = s._step(s)
        out[i] = s
    s._state = s
    return out
end

function generate_bitstream(s::TentMapRNGState, p, length)
    vals = s.random(length)
    return (vals < p).astype(np.uint8)
end

function reset(s::TentMapRNGState, x)
    s._state = x if x is ! nothing else s.x
    for _ in 1:s.burn_in
        s._state = s._step(s._state)
end

function state(s::TentMapRNGState)
    return s._state
end

end # module RngAccel
