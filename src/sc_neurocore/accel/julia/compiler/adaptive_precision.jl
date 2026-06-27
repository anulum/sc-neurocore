# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia validation mirror for compiler/adaptive_precision

module AdaptivePrecisionAccel

export LayerPrecisionState,
    SynapsePrecisionState,
    to_dict,
    validate_adaptive_precision,
    validate_synapse_precision

struct LayerPrecisionState
    layer_index::Int
    name::String
    bitstream_length::Int
    error_bound::Float64
    sensitivity::Float64

    function LayerPrecisionState(
        layer_index::Int,
        name::String,
        bitstream_length::Int,
        error_bound::Real,
        sensitivity::Real,
    )
        layer_index >= 0 || throw(ArgumentError("layer_index must be non-negative"))
        !isempty(name) || throw(ArgumentError("name must be a non-empty string"))
        bitstream_length > 0 || throw(ArgumentError("bitstream_length must be positive"))
        ispow2(bitstream_length) || throw(ArgumentError("bitstream_length must be a power of two"))
        error_scalar = _non_negative(error_bound, "error_bound")
        sensitivity_scalar = _non_negative(sensitivity, "sensitivity")
        new(layer_index, name, bitstream_length, error_scalar, sensitivity_scalar)
    end
end

struct SynapsePrecisionState
    layer_index::Int
    layer_name::String
    output_index::Int
    input_index::Int
    bit_width::Int
    bitstream_length::Int
    sensitivity::Float64
    quantization_error_bound::Float64
    stochastic_error_bound::Float64
    total_error_bound::Float64

    function SynapsePrecisionState(
        layer_index::Int,
        layer_name::String,
        output_index::Int,
        input_index::Int,
        bit_width::Int,
        bitstream_length::Int,
        sensitivity::Real,
        quantization_error_bound::Real,
        stochastic_error_bound::Real,
        total_error_bound::Real,
    )
        layer_index >= 0 || throw(ArgumentError("layer_index must be non-negative"))
        !isempty(layer_name) || throw(ArgumentError("layer_name must be a non-empty string"))
        output_index >= 0 || throw(ArgumentError("output_index must be non-negative"))
        input_index >= 0 || throw(ArgumentError("input_index must be non-negative"))
        bit_width > 0 || throw(ArgumentError("bit_width must be positive"))
        bitstream_length > 0 || throw(ArgumentError("bitstream_length must be positive"))
        sens = _non_negative(sensitivity, "sensitivity")
        q_bound = _non_negative(quantization_error_bound, "quantization_error_bound")
        s_bound = _non_negative(stochastic_error_bound, "stochastic_error_bound")
        total = _non_negative(total_error_bound, "total_error_bound")
        total + 1e-15 >= q_bound + s_bound ||
            throw(ArgumentError("total_error_bound must cover component bounds"))
        new(
            layer_index,
            layer_name,
            output_index,
            input_index,
            bit_width,
            bitstream_length,
            sens,
            q_bound,
            s_bound,
            total,
        )
    end
end

function to_dict(row::LayerPrecisionState)
    Dict(
        "layer_index" => row.layer_index,
        "name" => row.name,
        "bitstream_length" => row.bitstream_length,
        "error_bound" => row.error_bound,
        "sensitivity" => row.sensitivity,
    )
end

function to_dict(row::SynapsePrecisionState)
    Dict(
        "layer_index" => row.layer_index,
        "layer_name" => row.layer_name,
        "output_index" => row.output_index,
        "input_index" => row.input_index,
        "bit_width" => row.bit_width,
        "bitstream_length" => row.bitstream_length,
        "sensitivity" => row.sensitivity,
        "quantization_error_bound" => row.quantization_error_bound,
        "stochastic_error_bound" => row.stochastic_error_bound,
        "total_error_bound" => row.total_error_bound,
    )
end

function validate_adaptive_precision(row::LayerPrecisionState)
    try
        LayerPrecisionState(row.layer_index, row.name, row.bitstream_length, row.error_bound, row.sensitivity)
        true
    catch
        false
    end
end

function validate_synapse_precision(row::SynapsePrecisionState)
    try
        SynapsePrecisionState(
            row.layer_index,
            row.layer_name,
            row.output_index,
            row.input_index,
            row.bit_width,
            row.bitstream_length,
            row.sensitivity,
            row.quantization_error_bound,
            row.stochastic_error_bound,
            row.total_error_bound,
        )
        true
    catch
        false
    end
end

function _non_negative(value::Real, name::String)
    scalar = Float64(value)
    isfinite(scalar) && scalar >= 0.0 ||
        throw(ArgumentError("$name must be finite and non-negative"))
    scalar
end

end # module AdaptivePrecisionAccel
