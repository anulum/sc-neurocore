# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Wilson 1999 polynomial cortical model (parity with wilson_hr.py)
#
# Build:
#   mojo build --emit shared-lib -o libwilsonhr.so wilson_hr.mojo
#
# Parity contract: `wilson_hr_simulate_c` reproduces
# `sc_neurocore.neurons.models.wilson_hr.WilsonHRNeuron.simulate`. The polynomial
# RHS is exact arithmetic; each product is rounded into its own variable before
# the following add/subtract so the compiler cannot contract a multiply-add into a
# single-rounding FMA — the one operation that diverges from the IEEE-754
# two-rounding path used by Python/Rust/Go/Julia. The hard voltage reset on each
# spike re-anchors the trajectory, so any residual single-ULP difference does not
# accumulate; the backend is validated per-step and on spike counts.
#
# Mojo FFI rule (per feedback_mojo_026_ffi_pattern): @export rejects parametric
# signatures, so the output trace buffer is passed as a raw `Int` address and the
# pointer is reconstructed inside. The caller allocates n+2 Float64 slots:
# [0, n) receive the v trace, index n the final v, index n+1 the final r.
#
# Reference: Wilson, H.R. (1999). J. Theor. Biol. 200:375-388.

from std.memory import UnsafePointer


@export
fn wilson_hr_simulate_c(
    v0: Float64,
    r0: Float64,
    tau_r: Float64,
    v_peak: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var v = v0
    var r = r0
    var spikes: Int64 = 0
    for t in range(n_steps):
        var dv1 = _dv(v, r, current)
        var dr1 = _dr(v, r, tau_r)
        var hd = 0.5 * dt
        var dv2 = _dv(v + hd * dv1, r + hd * dr1, current)
        var dr2 = _dr(v + hd * dv1, r + hd * dr1, tau_r)
        var dv3 = _dv(v + hd * dv2, r + hd * dr2, current)
        var dr3 = _dr(v + hd * dv2, r + hd * dr2, tau_r)
        var dv4 = _dv(v + dt * dv3, r + dt * dr3, current)
        var dr4 = _dr(v + dt * dv3, r + dt * dr3, tau_r)
        var sv = dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4
        var sr = dr1 + 2.0 * dr2 + 2.0 * dr3 + dr4
        v = v + dt * sv / 6.0
        r = r + dt * sr / 6.0
        if v >= v_peak:
            v = -0.7
            spikes += 1
        trace[t] = v
    trace[n_steps] = v
    trace[n_steps + 1] = r
    return spikes


@always_inline
fn _dv(v: Float64, r: Float64, current: Float64) -> Float64:
    var quad = 32.63 * v * v
    var lin = 47.71 * v
    var poly = -(17.81 + lin + quad) * (v - 0.55)
    var syn = -26.0 * r * (v + 0.92)
    return poly + syn + current


@always_inline
fn _dr(v: Float64, r: Float64, tau_r: Float64) -> Float64:
    var num = -r + 1.35 * v + 1.03
    return num / tau_r
