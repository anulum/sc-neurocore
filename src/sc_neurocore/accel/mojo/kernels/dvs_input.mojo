# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo validation shim for interfaces/dvs_input

# This shim is deliberately narrow: the Python DVSInputLayer owns the event
# surface update and stochastic frame generation, while Mojo exposes
# FFI-checkable validation helpers for generated-kernel dispatchers.


def _finite(x: Float64) -> Bool:
    return x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308


@export
def dvs_input_params_valid_c(height: Int, width: Int, decay_tau: Float64) -> Int64:
    if height <= 0:
        return 0
    if width <= 0:
        return 0
    if not (_finite(decay_tau) and decay_tau > 0.0):
        return 0
    return 1


@export
def dvs_input_timestamp_valid_c(
    timestamp_ms: Float64,
    previous_timestamp_ms: Float64,
    has_previous: Int,
    last_update_time: Float64,
) -> Int64:
    if not (_finite(timestamp_ms) and _finite(last_update_time)):
        return 0
    if last_update_time < 0.0:
        return 0
    if has_previous != 0 and timestamp_ms < previous_timestamp_ms:
        return 0
    if timestamp_ms < last_update_time:
        return 0
    return 1


@export
def dvs_input_polarity_valid_c(polarity: Int) -> Int64:
    if polarity == -1 or polarity == 0 or polarity == 1:
        return 1
    return 0


@export
def dvs_input_coordinate_status_c(x: Int, y: Int, height: Int, width: Int) -> Int64:
    if height <= 0 or width <= 0:
        return -1
    if x < 0 or y < 0:
        return 0
    if x >= width or y >= height:
        return 0
    return 1


@export
def dvs_input_bitstream_length_valid_c(length: Int) -> Int64:
    if length > 0:
        return 1
    return 0
