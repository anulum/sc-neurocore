# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo validation shim for the L6 holonomic planetary adapter

# This shim is deliberately narrow: the Python adapter owns the traced Gaia-field
# update, while Mojo provides an FFI-checkable validation/projection contract for
# downstream generated-kernel dispatchers. It replaces the old non-parsing
# generated pseudo-code stub.


def _finite(x: Float64) -> Bool:
    return x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308


@export
def l6_plan_params_valid_c(
    n_regions: Int,
    bitstream_length: Int,
    f_schumann: Float64,
    q_factor: Float64,
    alpha_gaia: Float64,
    p_percolation: Float64,
) -> Int64:
    if n_regions <= 0:
        return 0
    if bitstream_length <= 0:
        return 0
    if not (_finite(f_schumann) and f_schumann > 0.0):
        return 0
    if not (_finite(q_factor) and q_factor > 0.0):
        return 0
    if not (_finite(alpha_gaia) and alpha_gaia > 0.0):
        return 0
    if not (_finite(p_percolation) and p_percolation > 0.0 and p_percolation < 1.0):
        return 0
    return 1


@export
def l6_plan_dt_valid_c(dt: Float64) -> Int64:
    if _finite(dt) and dt > 0.0:
        return 1
    return 0


@export
def l6_plan_input_shape_valid_c(
    input_rows: Int,
    input_cols: Int,
    bitstream_length: Int,
) -> Int64:
    if input_rows <= 0:
        return 0
    if bitstream_length <= 0:
        return 0
    if input_cols != bitstream_length:
        return 0
    return 1


@export
def l6_plan_projected_region_count_c(input_rows: Int, n_regions: Int) -> Int64:
    if input_rows <= 0 or n_regions <= 0:
        return -1
    return Int64(n_regions)
