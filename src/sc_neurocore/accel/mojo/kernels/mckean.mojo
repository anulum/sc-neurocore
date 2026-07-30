# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — source McKean Heaviside RK4 kernel
def _mckean_rhs_v(
    v: Float64, w: Float64, a: Float64, l: Float64, mu: Float64, current: Float64
) -> Float64:
    return -l * v + mu * (1.0 if v >= a else 0.0) - w + current


def _mckean_rhs_w(v: Float64, b: Float64) -> Float64:
    return b * v


def mckean_next_v(
    v: Float64,
    w: Float64,
    a: Float64,
    l: Float64,
    mu: Float64,
    b: Float64,
    dt: Float64,
    current: Float64,
) -> Float64:
    """Return the simultaneous RK4 voltage candidate."""
    var k1v = _mckean_rhs_v(v, w, a, l, mu, current)
    var k1w = _mckean_rhs_w(v, b)
    var k2v = _mckean_rhs_v(v + dt * k1v / 2.0, w + dt * k1w / 2.0, a, l, mu, current)
    var k2w = _mckean_rhs_w(v + dt * k1v / 2.0, b)
    var k3v = _mckean_rhs_v(v + dt * k2v / 2.0, w + dt * k2w / 2.0, a, l, mu, current)
    var k3w = _mckean_rhs_w(v + dt * k2v / 2.0, b)
    var k4v = _mckean_rhs_v(v + dt * k3v, w + dt * k3w, a, l, mu, current)
    return v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)


def mckean_next_w(
    v: Float64,
    w: Float64,
    a: Float64,
    l: Float64,
    mu: Float64,
    b: Float64,
    dt: Float64,
    current: Float64,
) -> Float64:
    """Return the simultaneous RK4 recovery candidate."""
    var k1v = _mckean_rhs_v(v, w, a, l, mu, current)
    var k1w = _mckean_rhs_w(v, b)
    var k2v = _mckean_rhs_v(v + dt * k1v / 2.0, w + dt * k1w / 2.0, a, l, mu, current)
    var k2w = _mckean_rhs_w(v + dt * k1v / 2.0, b)
    var k3v = _mckean_rhs_v(v + dt * k2v / 2.0, w + dt * k2w / 2.0, a, l, mu, current)
    var k3w = _mckean_rhs_w(v + dt * k2v / 2.0, b)
    var k4w = _mckean_rhs_w(v + dt * k3v, b)
    return w + (dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)
