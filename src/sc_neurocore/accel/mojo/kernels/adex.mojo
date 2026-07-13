# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo AdEx baseline-Euler simulator
#
# Build:
#   mojo build --emit shared-lib -o libadex.so adex.mojo
#
# The C ABI reproduces the maintained candidate-first explicit-Euler recurrence
# in sc_neurocore.neurons.models.adex.AdExNeuron. The caller supplies n_steps+2
# Float64 slots: the post-reset voltage trace followed by final (v, w). A
# negative return rejects invalid input or a non-finite candidate; the Python
# dispatcher never commits output state after rejection.
#
# Reference: Brette, R. & Gerstner, W. (2005). J. Neurophysiol. 94:3637-3642.
# DOI: 10.1152/jn.00686.2005.

from std.math import exp, isfinite
from std.memory import UnsafePointer


@always_inline
def _clip(value: Float64, lower: Float64, upper: Float64) -> Float64:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


@always_inline
def _state_valid(v: Float64, w: Float64) -> Bool:
    return isfinite(v) and isfinite(w)


struct AdEx(Copyable, Movable):
    var v_rest: Float64
    var v_reset: Float64
    var v_threshold: Float64
    var v_rh: Float64
    var delta_t: Float64
    var tau: Float64
    var tau_w: Float64
    var a: Float64
    var b: Float64
    var c_m: Float64
    var dt: Float64

    def __init__(
        out self,
        v_rest: Float64,
        v_reset: Float64,
        v_threshold: Float64,
        v_rh: Float64,
        delta_t: Float64,
        tau: Float64,
        tau_w: Float64,
        a: Float64,
        b: Float64,
        c_m: Float64,
        dt: Float64,
    ):
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.v_rh = v_rh
        self.delta_t = delta_t
        self.tau = tau
        self.tau_w = tau_w
        self.a = a
        self.b = b
        self.c_m = c_m
        self.dt = dt

    @always_inline
    def parameters_valid(self) -> Bool:
        return (
            isfinite(self.v_rest)
            and isfinite(self.v_reset)
            and isfinite(self.v_threshold)
            and isfinite(self.v_rh)
            and isfinite(self.delta_t)
            and isfinite(self.tau)
            and isfinite(self.tau_w)
            and isfinite(self.a)
            and isfinite(self.b)
            and isfinite(self.c_m)
            and isfinite(self.dt)
            and self.delta_t > 0.0
            and self.tau > 0.0
            and self.tau_w > 0.0
            and self.c_m > 0.0
            and self.dt > 0.0
        )


@export
def adex_simulate_c(
    v0: Float64,
    w0: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    v_rh: Float64,
    delta_t: Float64,
    tau: Float64,
    tau_w: Float64,
    a: Float64,
    b: Float64,
    c_m: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    output_addr: Int,
) -> Int64:
    if n_steps < 0 or output_addr == 0 or not isfinite(current):
        return -1

    var model = AdEx(
        v_rest,
        v_reset,
        v_threshold,
        v_rh,
        delta_t,
        tau,
        tau_w,
        a,
        b,
        c_m,
        dt,
    )
    var v = v0
    var w = w0
    if not model.parameters_valid() or not _state_valid(v, w):
        return -1

    var output = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=output_addr)
    var spikes: Int64 = 0
    for step in range(n_steps):
        var argument = _clip((v - model.v_rh) / model.delta_t, -20.0, 20.0)
        var exp_term = model.delta_t * exp(argument)
        var dv = (
            (-(v - model.v_rest) + exp_term) / model.tau
            + (-w + current) / model.c_m
        )
        var dw = (model.a * (v - model.v_rest) - w) / model.tau_w
        var next_v = v + dv * model.dt
        var next_w = w + dw * model.dt
        if not (
            isfinite(exp_term)
            and isfinite(dv)
            and isfinite(dw)
            and _state_valid(next_v, next_w)
        ):
            return -1
        if next_v >= model.v_threshold:
            var spike_w = next_w + model.b
            if not isfinite(spike_w):
                return -1
            v = model.v_reset
            w = spike_w
            spikes += 1
        else:
            v = next_v
            w = next_w
        output[step] = v

    output[n_steps] = v
    output[n_steps + 1] = w
    return spikes
