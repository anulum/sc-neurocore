# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Ermentrout-Kopell theta map (parity with ermentrout_kopell_map_neuron.py)
#
# Build:
#   mojo build --emit shared-lib -o libermentrout.so ermentrout_kopell_map_neuron.mojo
#
# Parity contract: `ermentrout_kopell_map_simulate_c` reproduces
# `sc_neurocore.neurons.models.ermentrout_kopell_map_neuron.ErmentroutKopellMapNeuron.simulate`.
# The only transcendental is `cos`, and the theta neuron is a non-chaotic phase
# oscillator, so Mojo's `cos` (which may differ from the reference libm by a ULP,
# and whose multiply-adds the release build may fuse) does not amplify: the trace
# stays within a small ULP band and the spike counts match. The wrap uses
# `theta - floor(theta / 2*pi) * 2*pi`, matching Python's `theta % (2*pi)`.
#
# Mojo FFI rule (per feedback_mojo_026_ffi_pattern): @export rejects parametric
# signatures, so the output trace buffer is passed as a raw `Int` address and the
# pointer is reconstructed inside. The caller allocates n+1 Float64 slots:
# [0, n) receive the theta trace, index n the final theta.
#
# Reference: Ermentrout & Kopell (1986) SIAM J Appl Math 46:233-253.

from std.memory import UnsafePointer
from std.math import cos, floor


@always_inline
def _fold_two_pi(v: Float64, two_pi: Float64) -> Float64:
    # Floored remainder modulo 2*pi: lands in [0, 2*pi), matching Python `%`.
    return v - floor(v / two_pi) * two_pi


@export
def ermentrout_kopell_map_simulate_c(
    theta0: Float64,
    dt: Float64,
    gain: Float64,
    theta_threshold: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var theta = theta0
    var inp = gain * current
    var two_pi = 2.0 * 3.141592653589793
    var spikes: Int64 = 0
    for t in range(n_steps):
        var theta_prev = theta
        var cos_theta = cos(theta)
        var d_theta = (1.0 - cos_theta) + (1.0 + cos_theta) * inp
        var theta_next = theta + dt * d_theta
        if theta_next >= theta_threshold and theta_prev < theta_threshold:
            spikes += 1
        theta = _fold_two_pi(theta_next, two_pi)
        trace[t] = theta
    if n_steps > 0:
        trace[n_steps] = theta
    return spikes
