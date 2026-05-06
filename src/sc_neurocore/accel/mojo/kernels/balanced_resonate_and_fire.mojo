# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for balanced_resonate_and_fire

from std.math import sqrt


fn sustain_oscillation_boundary(omega: Float64, dt: Float64) -> Float64:
    # Mirrors p(omega) = (-1 + sqrt(1 - (dt * omega)^2)) / dt.
    var scaled = dt * omega
    var radicand = 1.0 - scaled * scaled
    if radicand < 0.0:
        radicand = 0.0
    return (-1.0 + sqrt(radicand)) / dt


fn step_scalar(
    x: Float64,
    y: Float64,
    q: Float64,
    omega: Float64,
    b_offset: Float64,
    threshold: Float64,
    gamma: Float64,
    dt: Float64,
    current: Float64,
) -> Int:
    var p_omega = sustain_oscillation_boundary(omega, dt)
    var b_t = p_omega - b_offset - q
    var theta_t = threshold + q
    var x_next = x + dt * (b_t * x - omega * y + current)
    var _y_next = y + dt * (omega * x + b_t * y)
    var spike = 0
    if x_next >= theta_t:
        spike = 1
    var _q_next = gamma * q + Float64(spike)
    return spike
