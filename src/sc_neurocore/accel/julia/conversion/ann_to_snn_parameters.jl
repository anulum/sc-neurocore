# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia owned dense IF parameters

"""Owned row-major output-by-input coefficients and a finite per-layer preload."""
struct DenseLayer
    inputs::Int
    outputs::Int
    weights::Vector{Float64}
    bias::Union{Nothing,Vector{Float64}}
    threshold::Float64
    initial_fraction::Float64

    """Validate geometry and domains before taking independent float64 coefficient copies."""
    function DenseLayer(inputs::Int, outputs::Int, weights::AbstractVector{<:Real},
                        bias::Union{Nothing,AbstractVector{<:Real}}, threshold::Real;
                        initial_fraction::Real=0.0, max_working_bytes::Int=256 << 20)
        working_limit(max_working_bytes)
        inputs > 0 && outputs > 0 && BigInt(inputs) * outputs == length(weights) ||
            throw(ArgumentError("invalid dense coefficient dimensions"))
        isnothing(bias) || length(bias) == outputs || throw(ArgumentError("invalid bias dimensions"))
        coefficients = BigInt(length(weights)) + (isnothing(bias) ? 0 : length(bias))
        16coefficients <= max_working_bytes || throw(OutOfMemoryError())
        theta, fraction = Float64(threshold), Float64(initial_fraction)
        isfinite(theta) && theta > 0 && isfinite(fraction) ||
            throw(ArgumentError("threshold and preload must be finite; threshold positive"))
        owned = Float64.(weights)
        owned_bias = isnothing(bias) ? nothing : Float64.(bias)
        all(isfinite, owned) && (isnothing(owned_bias) || all(isfinite, owned_bias)) ||
            throw(ArgumentError("dense coefficients must be finite float64"))
        new(inputs, outputs, owned, owned_bias, theta, fraction)
    end
end

"""Connected owned dense layers with either final IF events or signed linear integration."""
struct ConvertedSNN
    layers::Vector{DenseLayer}
    output_mode::Symbol

    """Own and validate a nonempty connected stack within a declared copy budget."""
    function ConvertedSNN(layers::AbstractVector{DenseLayer}; output_mode::Symbol=:spikes,
                          max_working_bytes::Int=256 << 20)
        working_limit(max_working_bytes)
        isempty(layers) && throw(ArgumentError("at least one dense layer required"))
        output_mode in (:spikes, :linear) || throw(ArgumentError("invalid output mode"))
        all(layers[i].outputs == layers[i + 1].inputs for i in 1:length(layers)-1) ||
            throw(ArgumentError("disconnected dense stack"))
        coefficients = sum(BigInt(length(l.weights)) + (isnothing(l.bias) ? 0 : length(l.bias)) for l in layers)
        16coefficients <= max_working_bytes || throw(OutOfMemoryError())
        owned = [DenseLayer(l.inputs, l.outputs, l.weights, l.bias, l.threshold;
                           initial_fraction=l.initial_fraction, max_working_bytes=max_working_bytes) for l in layers]
        new(owned, output_mode)
    end
end
