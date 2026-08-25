# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Bertram et al. four-state RK4 kernel

from std.math import exp

def bertram_boltz(v: Float64, midpoint: Float64, slope: Float64) -> Float64:
    return 1.0 / (1.0 + exp((midpoint - v) / slope))

def bertram_rhs(v: Float64, n: Float64, s1: Float64, s2: Float64,
               current: Float64) -> Tuple[Float64, Float64, Float64, Float64]:
    var m_inf = bertram_boltz(v, -22.0, 7.5)
    var n_inf = bertram_boltz(v, -9.0, 10.0)
    var s1_inf = bertram_boltz(v, -40.0, 0.5)
    var s2_inf = bertram_boltz(v, -42.0, 0.4)
    var tau_n = 9.09 / (1.0 + exp((v + 9.0) / 10.0))
    var i_ca = 280.0 * m_inf * (v - 100.0)
    var i_k = 1300.0 * n * (v + 80.0)
    var i_s1 = 20.0 * s1 * (v + 80.0)
    var i_s2 = 32.0 * s2 * (v + 80.0)
    var i_l = 25.0 * (v + 40.0)
    return ((-i_ca-i_k-i_s1-i_s2-i_l+current)/4524.0,
            1.1*(n_inf-n)/tau_n, (s1_inf-s1)/1000.0,
            (s2_inf-s2)/120000.0)

def bertram_phantom_step(v: Float64, n: Float64, s1: Float64, s2: Float64,
                        current: Float64) -> Tuple[Float64, Float64, Float64, Float64, Int]:
    var k1v, k1n, k1s1, k1s2 = bertram_rhs(v, n, s1, s2, current)
    var k2v, k2n, k2s1, k2s2 = bertram_rhs(v+0.25*k1v, n+0.25*k1n,
                                            s1+0.25*k1s1, s2+0.25*k1s2, current)
    var k3v, k3n, k3s1, k3s2 = bertram_rhs(v+0.25*k2v, n+0.25*k2n,
                                            s1+0.25*k2s1, s2+0.25*k2s2, current)
    var k4v, k4n, k4s1, k4s2 = bertram_rhs(v+0.5*k3v, n+0.5*k3n,
                                            s1+0.5*k3s1, s2+0.5*k3s2, current)
    var scale = 0.5 / 6.0
    var next_v = v + scale*(k1v+2.0*k2v+2.0*k3v+k4v)
    var next_n = n + scale*(k1n+2.0*k2n+2.0*k3n+k4n)
    var next_s1 = s1 + scale*(k1s1+2.0*k2s1+2.0*k3s1+k4s1)
    var next_s2 = s2 + scale*(k1s2+2.0*k2s2+2.0*k3s2+k4s2)
    var event = Int(next_v >= -20.0 and v < -20.0)
    return (next_v, next_n, next_s1, next_s2, event)

def main():
    var v = -43.0
    var n = 0.03
    var s1 = 0.1
    var s2 = 0.434
    var events = 0
    for _ in range(10000):
        var event: Int
        v, n, s1, s2, event = bertram_phantom_step(v, n, s1, s2, 0.0)
        events += event
    print(v, n, s1, s2, events)
