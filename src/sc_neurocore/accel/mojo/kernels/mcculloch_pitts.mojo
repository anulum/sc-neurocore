# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-faithful McCulloch-Pitts Mojo kernel
#
# Build: mojo build --Werror --target-cpu x86-64-v3 --emit shared-lib -o libmcculloch_pitts.so mcculloch_pitts.mojo

from std.memory import Pointer


@always_inline
def _valid_count(value: Int) -> Bool:
    return value >= 0 and value <= 2147483647


@always_inline
def mcculloch_pitts_step(
    excitatory_count: Int,
    inhibitory_active: Bool,
    theta: Int,
) -> Int:
    if theta <= 0 or theta > 2147483647 or not _valid_count(excitatory_count):
        return -1
    if inhibitory_active:
        return 0
    if excitatory_count >= theta:
        return 1
    return 0


def _run_mcculloch_pitts(
    theta: Int,
    counts_addr: Int,
    flags_addr: Int,
    n_rows: Int,
    output_addr: Int,
    write_output: Bool,
) -> Int64:
    if theta <= 0 or theta > 2147483647 or n_rows < 0:
        return -1
    var counts = Pointer[Int64, MutAnyOrigin](unsafe_from_address=counts_addr)
    var flags = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=flags_addr)
    var output = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=output_addr)
    var event_count: Int64 = 0
    for index in range(n_rows):
        var count = Int(counts[unsafe_offset=index])
        var flag = Int(flags[unsafe_offset=index])
        if not _valid_count(count) or (flag != 0 and flag != 1):
            return -1
        var event = mcculloch_pitts_step(count, flag == 1, theta)
        if event < 0:
            return -1
        event_count += Int64(event)
        if write_output:
            output[unsafe_offset=index] = UInt8(event)
    return event_count


# The caller retains ownership of stable Int64 counts and UInt8 inhibitory flags
# and a disjoint writable UInt8 event buffer, each with n_rows elements.
# Counts lie in [0, 2147483647], flags in {0, 1}, theta in [1, 2147483647].
# Return the active-event count; -1 rejects the complete batch before output
# mutation. Zero rows return 0 without dereferencing or requiring any buffer.
@export
def mcculloch_pitts_evaluate_c(
    theta: Int,
    counts_addr: Int,
    flags_addr: Int,
    n_rows: Int,
    output_addr: Int,
) abi("C") -> Int64:
    if theta <= 0 or theta > 2147483647 or n_rows < 0:
        return -1
    if n_rows == 0:
        return 0
    if counts_addr == 0 or flags_addr == 0 or output_addr == 0:
        return -1
    var validated = _run_mcculloch_pitts(
        theta,
        counts_addr,
        flags_addr,
        n_rows,
        output_addr,
        False,
    )
    if validated < 0:
        return -1
    return _run_mcculloch_pitts(
        theta,
        counts_addr,
        flags_addr,
        n_rows,
        output_addr,
        True,
    )
