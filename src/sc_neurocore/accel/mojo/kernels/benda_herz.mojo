# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Benda-Herz adaptation kernel

from std.math import sqrt

fn benda_herz_rate(a: Float64, current: Float64, onset_gain: Float64,
                   rheobase: Float64) -> Float64:
    var drive = max(current - a - rheobase, 0.0)
    return onset_gain * sqrt(drive)

fn benda_herz_step(a: Float64, phase: Float64, current: Float64,
                   onset_gain: Float64, rheobase: Float64,
                   adaptation_slope: Float64, tau_a: Float64,
                   dt: Float64) -> Tuple[Float64, Float64, Int]:
    """Return the source RK4 adaptation, phase, and deterministic event."""
    var r1 = benda_herz_rate(a, current, onset_gain, rheobase)
    var k1a = (adaptation_slope*r1-a)/tau_a
    var a2 = a + 0.5*dt*k1a
    var r2 = benda_herz_rate(a2, current, onset_gain, rheobase)
    var k2a = (adaptation_slope*r2-a2)/tau_a
    var a3 = a + 0.5*dt*k2a
    var r3 = benda_herz_rate(a3, current, onset_gain, rheobase)
    var k3a = (adaptation_slope*r3-a3)/tau_a
    var a4 = a + dt*k3a
    var r4 = benda_herz_rate(a4, current, onset_gain, rheobase)
    var k4a = (adaptation_slope*r4-a4)/tau_a
    var scale = dt / 6.0
    var next_a = a + scale*(k1a + 2.0*k2a + 2.0*k3a + k4a)
    var next_phase = phase + scale*(r1 + 2.0*r2 + 2.0*r3 + r4)/1000.0
    if next_phase >= 1.0:
        return (next_a, 0.0, 1)
    return (next_a, next_phase, 0)
