# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo validation shim for the L13 holonomic source adapter

# This shim is deliberately narrow: the Python adapter owns the traced source-field
# update, while Mojo provides an FFI-checkable validation/projection contract for
# downstream generated-kernel dispatchers. It replaces the old non-parsing
# generated pseudo-code stub.


def _finite(x: Float64) -> Bool:
    return x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308


@export
def l13_source_params_valid_c(
    n_vacuum_nodes: Int,
    bitstream_length: Int,
    j_primordial_coupling: Float64,
    h_potential_bias: Float64,
    lambda_scission: Float64,
) -> Int64:
    if n_vacuum_nodes <= 0:
        return 0
    if bitstream_length <= 0:
        return 0
    if not _finite(j_primordial_coupling):
        return 0
    if not _finite(h_potential_bias):
        return 0
    if not (_finite(lambda_scission) and lambda_scission >= 0.0):
        return 0
    return 1


@export
def l13_source_dt_valid_c(dt: Float64) -> Int64:
    if _finite(dt) and dt > 0.0:
        return 1
    return 0


@export
def l13_source_feedback_shape_valid_c(input_rank: Int, input_rows: Int, input_cols: Int) -> Int64:
    if input_rank < 0 or input_rank > 2:
        return 0
    if input_rank == 0:
        return 1
    if input_rank == 1:
        if input_rows <= 0:
            return 0
        return 1
    if input_rows <= 0:
        return 0
    if input_cols <= 0:
        return 0
    return 1


@export
def l13_source_projected_node_count_c(input_rank: Int, input_rows: Int, n_vacuum_nodes: Int) -> Int64:
    if n_vacuum_nodes <= 0:
        return -1
    if input_rank < 0 or input_rank > 2:
        return -1
    if input_rank == 0:
        return Int64(n_vacuum_nodes)
    if input_rows <= 0:
        return -1
    return Int64(n_vacuum_nodes)
