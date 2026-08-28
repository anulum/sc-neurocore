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
# two-rounding path used by Python/Rust/Go/Julia. The backend is validated on
# its complete continuous trajectory and sampled upward-crossing count.
#
# Mojo FFI rule (per feedback_mojo_026_ffi_pattern): @export rejects parametric
# signatures, so the output trace buffer is passed as a raw `Int` address and the
# pointer is reconstructed inside. The caller allocates n+2 Float64 slots:
# [0, n) receive the v trace, index n the final v, index n+1 the final r.
#
# Reference: Wilson, H.R. (1999). J. Theor. Biol. 200:375-388.

from std.math import isfinite
from std.memory import UnsafePointer


@export
def wilson_hr_simulate_c(
    v0: Float64,
    r0: Float64,
    capacitance: Float64,
    tau_r: Float64,
    v_peak: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    if (
        n_steps < 0
        or trace_addr == 0
        or not isfinite(v0)
        or not isfinite(r0)
        or not isfinite(capacitance)
        or not isfinite(tau_r)
        or not isfinite(v_peak)
        or not isfinite(dt)
        or not isfinite(current)
        or capacitance <= 0.0
        or tau_r <= 0.0
        or dt <= 0.0
    ):
        return -1
    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var v = v0
    var r = r0
    var spikes: Int64 = 0
    for t in range(n_steps):
        var v_previous = v
        var dv1 = _dv(v, r, capacitance, current)
        var dr1 = _dr(v, r, tau_r)
        var hd = 0.5 * dt
        var dv2 = _dv(v + hd * dv1, r + hd * dr1, capacitance, current)
        var dr2 = _dr(v + hd * dv1, r + hd * dr1, tau_r)
        var dv3 = _dv(v + hd * dv2, r + hd * dr2, capacitance, current)
        var dr3 = _dr(v + hd * dv2, r + hd * dr2, tau_r)
        var dv4 = _dv(v + dt * dv3, r + dt * dr3, capacitance, current)
        var dr4 = _dr(v + dt * dv3, r + dt * dr3, tau_r)
        if (
            not isfinite(dv1)
            or not isfinite(dr1)
            or not isfinite(dv2)
            or not isfinite(dr2)
            or not isfinite(dv3)
            or not isfinite(dr3)
            or not isfinite(dv4)
            or not isfinite(dr4)
        ):
            return -1
        var sv = dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4
        var sr = dr1 + 2.0 * dr2 + 2.0 * dr3 + dr4
        var next_v = v + dt * sv / 6.0
        var next_r = r + dt * sr / 6.0
        if not isfinite(next_v) or not isfinite(next_r):
            return -1
        v = next_v
        r = next_r
        if v >= v_peak and v_previous < v_peak:
            spikes += 1
        trace[t] = v
    trace[n_steps] = v
    trace[n_steps + 1] = r
    return spikes


@always_inline
def _dv(v: Float64, r: Float64, capacitance: Float64, current: Float64) -> Float64:
    var quad = 32.63 * v * v
    var lin = 47.71 * v
    var poly = -(17.81 + lin + quad) * (v - 0.55)
    var syn = -26.0 * r * (v + 0.92)
    return (poly + syn + current) / capacitance


@always_inline
def _dr(v: Float64, r: Float64, tau_r: Float64) -> Float64:
    var num = -r + 1.35 * v + 1.03
    return num / tau_r
