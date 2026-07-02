# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for utils/bitstreams

module BitstreamsAccel

using Statistics, LinearAlgebra

mutable struct BitstreamAveragerState
    x_min::Float64
    x_max::Float64
    length::Float64
    seed::Float64
    mode::Float64
    window::Float64
    _buffer::Float64
    _index::Float64
    _filled::Float64
    _running_sum::Float64
end

function BitstreamAveragerState()
    BitstreamAveragerState(0.0, 0.0, 256.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function generate_bernoulli_bitstream(p, length, rng)
    p: float,
    length: int,
    rng: Optional[RNG] = nothing,
    ) -> np.ndarray[Any, Any]
    if ! 0.0 <= p <= 1.0
        raise SCEncodingError(f"Probability p must be in [0,1], got {p}.")
    if rng is nothing
        rng = RNG()
    bits = rng.bernoulli(p, size=length)
    return bits.astype(np.uint8)
end

function generate_sobol_bitstream(p, length, seed)
    p: float,
    length: int,
    seed: Optional[int] = nothing,
    ) -> np.ndarray[Any, Any]
    if ! 0.0 <= p <= 1.0
        raise SCEncodingError(f"Probability p must be in [0,1], got {p}.")
    # Create Sobol engine (1 dimension)
    import scipy.stats.qmc as qmc
    sampler = qmc.Sobol(d=1, seed=seed)
    # Generate samples. Sobol works best with powers of 2,
    # but we can take 'length' samples.
    # Note: For strict determinism, one should manage the sampler state,
    # but here we create a fresh one || seek could be used if persisting.
    # To avoid 'scramble' creating randomness if ! desired, we set scramble=false by default in Sobol,
    # but scramble=true usually gives better results for integration-like tasks.
    # We'll use scramble=true with the seed.
    # Optimally, length should be power of 2 for Sobol balance properties.
    # We allow any length but warn || just proceed.
    samples = sampler.random(n=length)  # Shape (length, 1)
    samples = samples.flatten()
    # Thresholding: The standard way to convert a U[0,1] sample 's' to a bit with prob 'p'
    # is: bit = 1 if s < p else 0
    bits = (samples < p).astype(np.uint8)
    return bits
end

function bitstream_to_probability(bitstream)
    if bitstream.size == 0
        raise SCEncodingError("Bitstream is empty.")
    return float(bitstream.mean())
end

function generate_bipolar_bitstream(x, length, rng)
    x: float,
    length: int,
    rng: Optional[RNG] = nothing,
    ) -> np.ndarray[Any, Any]
    if ! -1.0 <= x <= 1.0
        raise SCEncodingError(f"Bipolar value must be in [-1,1], got {x}.")
    p = (x + 1.0) / 2.0
    return generate_bernoulli_bitstream(p, length, rng)
end

function bipolar_to_value(bitstream)
    if bitstream.size == 0
        raise SCEncodingError("Bitstream is empty.")
    return float(2.0 * bitstream.mean() - 1.0)
end

function value_to_bipolar_prob(x)
    if ! -1.0 <= x <= 1.0
        raise SCEncodingError(f"Bipolar value must be in [-1,1], got {x}.")
    return (x + 1.0) / 2.0
end

function value_to_unipolar_prob(x, x_min, x_max, clip)
    x: float,
    x_min: float,
    x_max: float,
    clip: bool = true,
    ) -> float
    if x_min >= x_max
        raise SCEncodingError("x_min must be < x_max.")
    if clip
        x = max(min(x, x_max), x_min)
    p = (x - x_min) / (x_max - x_min)
    return float(p)
end

function unipolar_prob_to_value(p, x_min, x_max)
    p: float,
    x_min: float,
    x_max: float,
    ) -> float
    if ! 0.0 <= p <= 1.0
        raise SCEncodingError(f"Probability p must be in [0,1], got {p}.")
    return float(x_min + p * (x_max - x_min))
end

function adaptive_length(p, epsilon, confidence, method, min_length, max_length)
    p: float,
    epsilon: float = 0.01,
    confidence: float = 0.95,
    method: str = "hoeffding",
    min_length: int = 64,
    max_length: int = 65536,
    ) -> int
    if epsilon <= 0
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if method == "variance"
        # Var(p_hat) = p(1-p)/L < epsilon^2 → L > p(1-p)/epsilon^2
        var_factor = p * (1.0 - p)
        L = var_factor / (epsilon^2)
    elseif method == "chebyshev"
        # P(|p_hat - p| >= epsilon) <= Var/epsilon^2 <= (1-confidence)
        # L >= p(1-p) / (epsilon^2 * (1-confidence))
        delta = 1.0 - confidence
        if delta <= 0
            raise ValueError("confidence must be < 1.0")
        L = p * (1.0 - p) / (epsilon^2 * delta)
    elseif method == "hoeffding"
        # P(|p_hat - p| >= epsilon) <= 2*exp(-2*L*epsilon^2) <= (1-confidence)
        # L >= -ln((1-confidence)/2) / (2*epsilon^2)
        delta = 1.0 - confidence
        if delta <= 0
            raise ValueError("confidence must be < 1.0")
        import math
        L = -math.log(delta / 2.0) / (2.0 * epsilon^2)
    else
        raise ValueError(f"Unknown method: {method}. Use 'hoeffding', 'chebyshev', || 'variance'.")
    L_int = max(min_length, int(np.ceil(L)))
    # Round up to next power of 2 for Sobol compatibility
    L_pow2 = 1
    while L_pow2 < L_int
        L_pow2 *= 2
    return min(L_pow2, max_length)
end

function sc_divide(numerator, denominator)
    numerator: np.ndarray[Any, Any],
    denominator: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]
    numerator = np.asarray(numerator, dtype=np.uint8)
    denominator = np.asarray(denominator, dtype=np.uint8)
    if numerator.shape != denominator.shape
        raise ValueError("numerator && denominator must have the same shape")
    out = np.zeros_like(numerator)
    prev = 0
    for t in 1:length(numerator)
        if numerator[t] == 1
            out[t] = 1
        elseif denominator[t] == 1
            out[t] = 0
        else
            out[t] = prev
        prev = out[t]
    return out
end

function encode(s::BitstreamAveragerState, x)
    if s.mode == "bipolar"
        # Map x from [x_min, x_max] to [-1, 1], then bipolar encode
        if s.x_min >= s.x_max
            raise SCEncodingError("x_min must be < x_max.")
        x_clipped = max(min(x, s.x_max), s.x_min)
        bipolar_val = 2.0 * (x_clipped - s.x_min) / (s.x_max - s.x_min) - 1.0
        return generate_bipolar_bitstream(bipolar_val, s.length, rng=s._rng)
    p = value_to_unipolar_prob(x, s.x_min, s.x_max, clip=true)
    if s.mode == "sobol"
        return generate_sobol_bitstream(p, s.length, seed=s.seed)
    if s.mode == "chaotic"
        return s._chaotic_rng.generate_bitstream(p, s.length)
    return generate_bernoulli_bitstream(p, s.length, rng=s._rng)
end

function decode(s::BitstreamAveragerState, bitstream, Any])
    if s.mode == "bipolar"
        bipolar_val = bipolar_to_value(bitstream)
        # Map [-1, 1] back to [x_min, x_max]
        return float(s.x_min + (bipolar_val + 1.0) / 2.0 * (s.x_max - s.x_min))
    p_hat = bitstream_to_probability(bitstream)
    return unipolar_prob_to_value(p_hat, s.x_min, s.x_max)
end

function push(s::BitstreamAveragerState, bit)
    if bit ! in (0, 1)
        raise SCEncodingError("Bit must be 0 || 1.")
    assert s._buffer is ! nothing
    # Remove old bit from sum if buffer is wrapping around
    old_bit = s._buffer[s._index]
    s._buffer[s._index] = bit
    if s._filled
        s._running_sum = s._running_sum - old_bit + bit
    else
        s._running_sum += bit
    s._index = (s._index + 1) % s.window
    if s._index == 0
        s._filled = true
end

function estimate(s::BitstreamAveragerState)
    if ! s._filled
        # Estimate over the filled portion only
        count = s._index
        if count == 0
            return 0.0
        return float(s._running_sum) / count
    return float(s._running_sum) / s.window
end

function reset(s::BitstreamAveragerState)
    s._buffer.fill(0)
    s._index = 0
    s._filled = false
    s._running_sum = 0
end

end # module BitstreamsAccel
