# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for mcculloch_pitts

fn _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


fn mcculloch_pitts_step(weighted_input: Float64, theta: Float64) -> Int:
    if not _finite(weighted_input) or not _finite(theta):
        return -1
    if weighted_input >= theta:
        return 1
    return 0


fn step(weighted_input: Int) -> Int:
    if weighted_input >= 1:
        return 1
    return 0

fn reset() -> Int:
    return 0
