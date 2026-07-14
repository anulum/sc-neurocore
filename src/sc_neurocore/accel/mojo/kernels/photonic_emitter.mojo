# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fail-closed photonic crosstalk C ABI
#
# Build:
#   mojo build --emit shared-lib -o libphotonic_emitter.so photonic_emitter.mojo
#
# Output triples are coupling coefficient per µm, coupling ratio, isolation dB.
# A negative return code denotes invalid input; validation occurs before writes.

from std.math import exp, isfinite, log, pi, sin, sqrt
from std.memory import UnsafePointer


comptime ISOLATION_CEILING_DB = 300.0
comptime ISOLATION_RATIO_FLOOR = 1.0e-15
comptime LN_TEN = 2.302585092994046


@always_inline
def _valid_material(
    wavelength_nm: Float64,
    core_index: Float64,
    cladding_index: Float64,
) -> Bool:
    return (
        isfinite(wavelength_nm)
        and wavelength_nm > 0.0
        and isfinite(core_index)
        and core_index > 0.0
        and isfinite(cladding_index)
        and cladding_index > 0.0
        and core_index > cladding_index
    )


@always_inline
def _valid_pair(
    gap_nm: Float64,
    coupling_length_um: Float64,
    wavelength_nm: Float64,
    core_index: Float64,
    cladding_index: Float64,
) -> Bool:
    return (
        isfinite(gap_nm)
        and gap_nm >= 0.0
        and isfinite(coupling_length_um)
        and coupling_length_um >= 0.0
        and _valid_material(wavelength_nm, core_index, cladding_index)
    )


@always_inline
def _write_pair(
    gap_nm: Float64,
    coupling_length_um: Float64,
    wavelength_nm: Float64,
    core_index: Float64,
    cladding_index: Float64,
    output: UnsafePointer[Float64, MutAnyOrigin],
    offset: Int,
):
    var contrast = sqrt(core_index * core_index - cladding_index * cladding_index)
    var decay_length_nm = wavelength_nm / (2.0 * pi * contrast)
    var effective_index_difference = 0.1 * exp(-gap_nm / decay_length_nm)
    var coefficient = pi * effective_index_difference / (wavelength_nm * 1.0e-3)
    var ratio = sin(coefficient * coupling_length_um) ** 2
    var isolation = ISOLATION_CEILING_DB
    if ratio >= ISOLATION_RATIO_FLOOR:
        isolation = -10.0 * log(ratio) / LN_TEN
    output[offset] = coefficient
    output[offset + 1] = ratio
    output[offset + 2] = isolation


@export
def photonic_crosstalk_pair_c(
    gap_nm: Float64,
    coupling_length_um: Float64,
    wavelength_nm: Float64,
    core_index: Float64,
    cladding_index: Float64,
    output_addr: Int,
) -> Int64:
    if output_addr == 0 or not _valid_pair(
        gap_nm,
        coupling_length_um,
        wavelength_nm,
        core_index,
        cladding_index,
    ):
        return -1
    var output = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=output_addr)
    _write_pair(
        gap_nm,
        coupling_length_um,
        wavelength_nm,
        core_index,
        cladding_index,
        output,
        0,
    )
    return 0


@export
def photonic_crosstalk_batch_c(
    gaps_addr: Int,
    lengths_addr: Int,
    pair_count: Int,
    wavelength_nm: Float64,
    core_index: Float64,
    cladding_index: Float64,
    output_addr: Int,
) -> Int64:
    if (
        gaps_addr == 0
        or lengths_addr == 0
        or output_addr == 0
        or pair_count < 0
        or not _valid_material(wavelength_nm, core_index, cladding_index)
    ):
        return -1
    var gaps = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=gaps_addr)
    var lengths = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=lengths_addr)
    for index in range(pair_count):
        if not _valid_pair(
            gaps[index],
            lengths[index],
            wavelength_nm,
            core_index,
            cladding_index,
        ):
            return -1

    var output = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=output_addr)
    for index in range(pair_count):
        _write_pair(
            gaps[index],
            lengths[index],
            wavelength_nm,
            core_index,
            cladding_index,
            output,
            3 * index,
        )
    return 0
