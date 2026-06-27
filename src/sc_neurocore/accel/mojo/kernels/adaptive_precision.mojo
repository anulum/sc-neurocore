# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo adaptive precision validation kernel

from std.math import isfinite


def is_power_of_two(value: Int) -> Bool:
    if value <= 0:
        return False
    var probe = value
    while probe % 2 == 0:
        probe = probe // 2
    return probe == 1


def validate_non_negative(value: Float64) -> Bool:
    return isfinite(value) and value >= 0.0


def validate_layer_precision(
    layer_index: Int,
    name_length: Int,
    bitstream_length: Int,
    error_bound: Float64,
    sensitivity: Float64,
) -> Bool:
    if layer_index < 0 or name_length <= 0:
        return False
    if not is_power_of_two(bitstream_length):
        return False
    return validate_non_negative(error_bound) and validate_non_negative(sensitivity)


def validate_synapse_precision(
    layer_index: Int,
    layer_name_length: Int,
    output_index: Int,
    input_index: Int,
    bit_width: Int,
    bitstream_length: Int,
    sensitivity: Float64,
    quantization_error_bound: Float64,
    stochastic_error_bound: Float64,
    total_error_bound: Float64,
) -> Bool:
    if layer_index < 0 or layer_name_length <= 0:
        return False
    if output_index < 0 or input_index < 0:
        return False
    if bit_width <= 0 or bitstream_length <= 0:
        return False
    if not validate_non_negative(sensitivity):
        return False
    if not validate_non_negative(quantization_error_bound):
        return False
    if not validate_non_negative(stochastic_error_bound):
        return False
    if not validate_non_negative(total_error_bound):
        return False
    return total_error_bound + 1.0e-15 >= quantization_error_bound + stochastic_error_bound


def validate_adaptive_precision_kernel() -> Bool:
    if not validate_layer_precision(0, 3, 256, 0.03125, 0.5):
        return False
    if validate_layer_precision(0, 3, 300, 0.03125, 0.5):
        return False
    if validate_layer_precision(-1, 3, 256, 0.03125, 0.5):
        return False
    if not validate_synapse_precision(0, 3, 1, 2, 8, 128, 0.5, 0.01, 0.02, 0.03):
        return False
    if validate_synapse_precision(-1, 3, 1, 2, 8, 128, 0.5, 0.01, 0.02, 0.03):
        return False
    return not validate_synapse_precision(0, 3, 1, 2, 8, 128, 0.5, 0.02, 0.02, 0.03)


def main() raises:
    if not validate_adaptive_precision_kernel():
        raise Error("adaptive precision validation failed")
