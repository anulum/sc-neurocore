# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo benchmark for balanced_resonate_and_fire

from std.math import sqrt
from time import perf_counter_ns


fn sustain_oscillation_boundary(omega: Float64, dt: Float64) -> Float64:
    var scaled = dt * omega
    var radicand = 1.0 - scaled * scaled
    if radicand < 0.0:
        radicand = 0.0
    return (-1.0 + sqrt(radicand)) / dt


fn run_benchmark():
    alias N_STEPS = 200000
    var x = 0.0
    var y = 0.0
    var q = 0.0
    var omega = 10.0
    var b_offset = 1.0
    var threshold = 1.0
    var gamma = 0.9
    var dt = 0.01
    var current = 2.0
    var spikes = 0
    var t0 = perf_counter_ns()
    for _ in range(N_STEPS):
        var p_omega = sustain_oscillation_boundary(omega, dt)
        var b_t = p_omega - b_offset - q
        var theta_t = threshold + q
        var x_prev = x
        var y_prev = y
        x = x_prev + dt * (b_t * x_prev - omega * y_prev + current)
        y = y_prev + dt * (omega * x_prev + b_t * y_prev)
        var spike = 0
        if x >= theta_t:
            spike = 1
        q = gamma * q + Float64(spike)
        spikes += spike
    var elapsed_ns = perf_counter_ns() - t0
    var step_ns = Float64(elapsed_ns) / Float64(N_STEPS)
    print("backend mojo")
    print("status executed")
    print("n_steps", N_STEPS)
    print("current", current)
    print("omega", omega)
    print("b_offset", b_offset)
    print("elapsed_seconds", Float64(elapsed_ns) / 1000000000.0)
    print("step_ns", step_ns)
    print("spikes", spikes)
    print("final_x", x)
    print("final_y", y)
    print("final_q", q)


fn main() raises:
    run_benchmark()
