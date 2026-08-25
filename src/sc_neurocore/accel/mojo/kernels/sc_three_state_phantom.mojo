# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo retained three-state phantom identity

from std.math import exp

def sc_three_state_boltz(v: Float64, midpoint: Float64, slope: Float64) -> Float64:
    return 1.0 / (1.0 + exp((midpoint - v) / slope))

def sc_three_state_rhs(v: Float64, s1: Float64, s2: Float64,
                      current: Float64) -> Tuple[Float64, Float64, Float64]:
    var m_inf = sc_three_state_boltz(v, -20.0, 12.0)
    var n_inf = sc_three_state_boltz(v, -16.0, 5.6)
    var s1_inf = sc_three_state_boltz(v, -40.0, 10.0)
    var s2_inf = sc_three_state_boltz(v, -42.0, 0.4)
    var i_ca = 3.6*m_inf*(v-25.0)
    var i_k = 10.0*n_inf*(v+75.0)
    var i_s1 = 4.0*s1*(v+75.0)
    var i_s2 = 4.0*s2*(v+75.0)
    var i_l = 0.2*(v+40.0)
    return ((-i_ca-i_k-i_s1-i_s2-i_l+current)/5.3,
            (s1_inf-s1)/20000.0, (s2_inf-s2)/100000.0)

def sc_three_state_phantom_step(v: Float64, s1: Float64, s2: Float64,
                               current: Float64) -> Tuple[Float64, Float64, Float64, Int]:
    var k1v, k1s1, k1s2 = sc_three_state_rhs(v, s1, s2, current)
    var k2v, k2s1, k2s2 = sc_three_state_rhs(v+0.25*k1v, s1+0.25*k1s1, s2+0.25*k1s2, current)
    var k3v, k3s1, k3s2 = sc_three_state_rhs(v+0.25*k2v, s1+0.25*k2s1, s2+0.25*k2s2, current)
    var k4v, k4s1, k4s2 = sc_three_state_rhs(v+0.5*k3v, s1+0.5*k3s1, s2+0.5*k3s2, current)
    var scale = 0.5/6.0
    var next_v = v+scale*(k1v+2.0*k2v+2.0*k3v+k4v)
    var next_s1 = s1+scale*(k1s1+2.0*k2s1+2.0*k3s1+k4s1)
    var next_s2 = s2+scale*(k1s2+2.0*k2s2+2.0*k3s2+k4s2)
    return (next_v, next_s1, next_s2, Int(next_v >= -20.0 and v < -20.0))

def main():
    var v = -50.0
    var s1 = 0.1
    var s2 = 0.1
    var events = 0
    for _ in range(10000):
        var event: Int
        v, s1, s2, event = sc_three_state_phantom_step(v, s1, s2, 0.0)
        events += event
    print(v, s1, s2, events)
