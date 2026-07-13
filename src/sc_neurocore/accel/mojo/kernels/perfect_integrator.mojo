# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Perfect Integrator sequential simulator
#
# Build:
#   mojo build --emit shared-lib -o libperfect_integrator.so perfect_integrator.mojo
#
# The caller supplies n_steps+1 Float64 slots: the post-step voltage trace and
# final voltage. Rejected contracts leave the buffer untouched because a
# validation pass completes before emission.

from std.math import isfinite
from std.memory import UnsafePointer


struct PerfectIntegrator(Copyable, Movable):
    var c_m: Float64
    var v_threshold: Float64
    var v_reset: Float64
    var dt: Float64

    def __init__(
        out self,
        c_m: Float64,
        v_threshold: Float64,
        v_reset: Float64,
        dt: Float64,
    ):
        self.c_m = c_m
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.dt = dt

    @always_inline
    def valid(self, v: Float64) -> Bool:
        return (
            isfinite(v)
            and isfinite(self.c_m)
            and self.c_m > 0.0
            and isfinite(self.v_threshold)
            and isfinite(self.v_reset)
            and self.v_threshold > self.v_reset
            and v < self.v_threshold
            and isfinite(self.dt)
            and self.dt > 0.0
        )


@always_inline
def _candidate(model: PerfectIntegrator, v: Float64, current: Float64) -> Float64:
    return v + current / model.c_m * model.dt


def perfect_integrator_valid(
    v: Float64,
    c_m: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    dt: Float64,
) -> Bool:
    return PerfectIntegrator(c_m, v_threshold, v_reset, dt).valid(v)


def perfect_integrator_step_spike(
    v: Float64,
    current: Float64,
    c_m: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    dt: Float64,
) -> Int:
    var model = PerfectIntegrator(c_m, v_threshold, v_reset, dt)
    if not isfinite(current) or not model.valid(v):
        return -1
    var next_v = _candidate(model, v, current)
    if not isfinite(next_v):
        return -1
    return Int(next_v >= v_threshold)


def _run_perfect_integrator(
    model: PerfectIntegrator,
    v0: Float64,
    n_steps: Int,
    current: Float64,
    output_addr: Int,
    write_output: Bool,
) -> Int64:
    var v = v0
    if not model.valid(v):
        return -1
    var output = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=output_addr
    )
    var spikes: Int64 = 0
    for index in range(n_steps):
        var next_v = _candidate(model, v, current)
        if not isfinite(next_v):
            return -1
        if next_v >= model.v_threshold:
            v = model.v_reset
            spikes += 1
        else:
            v = next_v
        if write_output:
            output[index] = v
    if write_output:
        output[n_steps] = v
    return spikes


@export
def perfect_integrator_simulate_c(
    v0: Float64,
    c_m: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    output_addr: Int,
) -> Int64:
    if n_steps < 0 or output_addr == 0 or not isfinite(current):
        return -1
    var model = PerfectIntegrator(c_m, v_threshold, v_reset, dt)
    var validated = _run_perfect_integrator(
        model, v0, n_steps, current, output_addr, False
    )
    if validated < 0:
        return -1
    return _run_perfect_integrator(
        model, v0, n_steps, current, output_addr, True
    )
