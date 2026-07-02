# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo validation shim for the L9 holonomic memory adapter

# This shim is deliberately narrow: the Python adapter owns the traced TSVF
# update, while Mojo provides an FFI-checkable validation/projection contract for
# downstream generated-kernel dispatchers. It replaces the old non-parsing
# generated pseudo-code stub.


def _finite(x: Float64) -> Bool:
    return x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308


@export
def l9_mem_params_valid_c(
    n_memory_slots: Int,
    bitstream_length: Int,
    retrieval_gain: Float64,
    weak_measurement_strength: Float64,
    temporal_window: Int,
) -> Int64:
    if n_memory_slots <= 0:
        return 0
    if bitstream_length <= 0:
        return 0
    if temporal_window <= 0:
        return 0
    if not (_finite(retrieval_gain) and retrieval_gain >= 0.0):
        return 0
    if not (
        _finite(weak_measurement_strength)
        and weak_measurement_strength >= 0.0
        and weak_measurement_strength <= 1.0
    ):
        return 0
    return 1


@export
def l9_mem_dt_valid_c(dt: Float64) -> Int64:
    if _finite(dt) and dt > 0.0:
        return 1
    return 0


@export
def l9_mem_input_shape_valid_c(
    input_rows: Int,
    input_cols: Int,
    bitstream_length: Int,
) -> Int64:
    if input_rows <= 0:
        return 0
    if input_cols != bitstream_length:
        return 0
    if bitstream_length <= 0:
        return 0
    return 1


@export
def l9_mem_project_slot_c(slot: Int, input_rows: Int) -> Int64:
    if slot < 0 or input_rows <= 0:
        return -1
    return Int64(slot % input_rows)
