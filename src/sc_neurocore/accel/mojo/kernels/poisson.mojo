# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for poisson

fn _finite(x: Float64) -> Bool:
    return (
        x == x and x <= 1.7976931348623157e308 and x >= -1.7976931348623157e308
    )


fn poisson_saturated_spike(rate_hz: Float64, rate_override: Float64, dt_ms: Float64) -> Int:
    if not _finite(rate_hz) or rate_hz < 0.0:
        return -1
    if not _finite(rate_override):
        return -1
    if not _finite(dt_ms) or dt_ms <= 0.0:
        return -1
    var active_rate = rate_hz
    if rate_override >= 0.0:
        active_rate = rate_override
    if not _finite(active_rate) or active_rate < 0.0:
        return -1
    var hazard = active_rate * dt_ms / 1000.0
    if not _finite(hazard) or hazard < 0.0:
        return -1
    if hazard >= 36.7368005696771:
        return 1
    return 0


fn step(rate_override: Int) -> Int:
    if rate_override >= 1000000000:
        return 1
    return 0

fn reset() -> Int:
    return 0
