# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Fourcaud-Trocmé ExpIF simulator
#
# Build:
#   mojo build --emit shared-lib -o libexpif.so expif.mojo
#
# The compatibility ABI accepts n_steps+2 Float64 slots. The complete ABI uses
# aligned voltage/refractory/event buffers plus final-state slots. Rejected
# contracts leave every caller buffer untouched because a validation pass
# completes before emission.

from std.math import exp, isfinite
from std.memory import UnsafePointer


struct ExpIF(Copyable, Movable):
    var v_rest: Float64
    var v_reset: Float64
    var v_threshold: Float64
    var v_rh: Float64
    var delta_t: Float64
    var tau: Float64
    var dt: Float64
    var refractory_period: Float64
    var source_profile: Bool

    def __init__(
        out self,
        v_rest: Float64,
        v_reset: Float64,
        v_threshold: Float64,
        v_rh: Float64,
        delta_t: Float64,
        tau: Float64,
        dt: Float64,
        refractory_period: Float64,
        source_profile: Bool,
    ):
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.v_rh = v_rh
        self.delta_t = delta_t
        self.tau = tau
        self.dt = dt
        self.refractory_period = refractory_period
        self.source_profile = source_profile

    @always_inline
    def parameters_valid(self) -> Bool:
        return (
            isfinite(self.v_rest)
            and isfinite(self.v_reset)
            and isfinite(self.v_threshold)
            and isfinite(self.v_rh)
            and isfinite(self.delta_t)
            and isfinite(self.tau)
            and isfinite(self.dt)
            and isfinite(self.refractory_period)
            and self.delta_t > 0.0
            and self.tau > 0.0
            and self.dt > 0.0
            and self.refractory_period >= 0.0
            and self.v_threshold > self.v_rh
            and self.v_rest < self.v_threshold
            and self.v_reset < self.v_threshold
            and (
                not self.source_profile
                or (
                    self.v_rest == -65.0
                    and self.v_reset == -68.0
                    and self.v_threshold == -30.0
                    and self.v_rh == -59.9
                    and self.delta_t == 3.48
                    and self.tau == 10.0
                    and self.dt < 0.02
                    and self.refractory_period == 1.7
                )
            )
        )


@always_inline
def _rhs(model: ExpIF, v: Float64, current: Float64) -> Float64:
    var bounded_v = v
    if bounded_v > model.v_threshold:
        bounded_v = model.v_threshold
    var exp_term = model.delta_t * exp((bounded_v - model.v_rh) / model.delta_t)
    return (-(bounded_v - model.v_rest) + exp_term + current) / model.tau


def _run_expif(
    model: ExpIF,
    v0: Float64,
    refractory0: Float64,
    n_steps: Int,
    current: Float64,
    output_addr: Int,
    write_output: Bool,
) -> Int64:
    var v = v0
    var refractory = refractory0
    if (
        not model.parameters_valid()
        or not isfinite(v)
        or v >= model.v_threshold
        or not isfinite(refractory)
        or refractory < 0.0
        or refractory > model.refractory_period
    ):
        return -1

    var output = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=output_addr)
    var spikes: Int64 = 0
    for index in range(n_steps):
        if refractory > 0.0:
            refractory -= model.dt
            if refractory < 0.0:
                refractory = 0.0
            v = model.v_reset
        else:
            var k1 = _rhs(model, v, current)
            var k2 = _rhs(model, v + 0.5 * model.dt * k1, current)
            var k3 = _rhs(model, v + 0.5 * model.dt * k2, current)
            var k4 = _rhs(model, v + model.dt * k3, current)
            var next_v = v + (model.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if not (
                isfinite(k1)
                and isfinite(k2)
                and isfinite(k3)
                and isfinite(k4)
                and isfinite(next_v)
            ):
                return -1
            if next_v >= model.v_threshold:
                v = model.v_reset
                refractory = model.refractory_period
                spikes += 1
            else:
                v = next_v
        if write_output:
            output[index] = v

    if write_output:
        output[n_steps] = v
        output[n_steps + 1] = refractory
    return spikes


@export
def expif_simulate_c(
    v0: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    v_rh: Float64,
    delta_t: Float64,
    tau: Float64,
    dt: Float64,
    refractory_period: Float64,
    refractory_remaining: Float64,
    n_steps: Int,
    current: Float64,
    output_addr: Int,
) -> Int64:
    if n_steps < 0 or output_addr == 0 or not isfinite(current):
        return -1
    var model = ExpIF(
        v_rest,
        v_reset,
        v_threshold,
        v_rh,
        delta_t,
        tau,
        dt,
        refractory_period,
        False,
    )
    var validated = _run_expif(
        model,
        v0,
        refractory_remaining,
        n_steps,
        current,
        output_addr,
        False,
    )
    if validated < 0:
        return -1
    return _run_expif(
        model,
        v0,
        refractory_remaining,
        n_steps,
        current,
        output_addr,
        True,
    )


def _run_expif_complete(
    model: ExpIF,
    v0: Float64,
    refractory0: Float64,
    n_steps: Int,
    current: Float64,
    voltage_addr: Int,
    refractory_addr: Int,
    event_addr: Int,
    write_output: Bool,
) -> Int64:
    var v = v0
    var refractory = refractory0
    if (
        not model.parameters_valid()
        or not isfinite(v)
        or v >= model.v_threshold
        or not isfinite(refractory)
        or refractory < 0.0
        or refractory > model.refractory_period
    ):
        return -1

    var voltage = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=voltage_addr)
    var refractory_trace = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=refractory_addr
    )
    var events = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=event_addr)
    var spikes: Int64 = 0
    for index in range(n_steps):
        var event: UInt8 = 0
        if refractory > 0.0:
            refractory -= model.dt
            if refractory < 0.0:
                refractory = 0.0
            v = model.v_reset
        else:
            var k1 = _rhs(model, v, current)
            var k2: Float64
            var k3: Float64 = 0.0
            var k4: Float64 = 0.0
            var next_v: Float64
            if model.source_profile:
                k2 = _rhs(model, v + model.dt * k1, current)
                next_v = v + 0.5 * model.dt * (k1 + k2)
            else:
                k2 = _rhs(model, v + 0.5 * model.dt * k1, current)
                k3 = _rhs(model, v + 0.5 * model.dt * k2, current)
                k4 = _rhs(model, v + model.dt * k3, current)
                next_v = v + (model.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if not (
                isfinite(k1)
                and isfinite(k2)
                and isfinite(k3)
                and isfinite(k4)
                and isfinite(next_v)
            ):
                return -1
            if next_v >= model.v_threshold:
                v = model.v_reset
                refractory = model.refractory_period
                event = 1
                spikes += 1
            else:
                v = next_v
        if write_output:
            voltage[index] = v
            refractory_trace[index] = refractory
            events[index] = event

    if write_output:
        voltage[n_steps] = v
        refractory_trace[n_steps] = refractory
    return spikes


@export
def expif_simulate_complete_c(
    v0: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    v_rh: Float64,
    delta_t: Float64,
    tau: Float64,
    dt: Float64,
    refractory_period: Float64,
    refractory_remaining: Float64,
    source_profile: Int,
    n_steps: Int,
    current: Float64,
    voltage_addr: Int,
    refractory_addr: Int,
    event_addr: Int,
) -> Int64:
    if (
        n_steps < 0
        or voltage_addr == 0
        or refractory_addr == 0
        or event_addr == 0
        or not isfinite(current)
    ):
        return -1
    var model = ExpIF(
        v_rest,
        v_reset,
        v_threshold,
        v_rh,
        delta_t,
        tau,
        dt,
        refractory_period,
        source_profile != 0,
    )
    var validated = _run_expif_complete(
        model,
        v0,
        refractory_remaining,
        n_steps,
        current,
        voltage_addr,
        refractory_addr,
        event_addr,
        False,
    )
    if validated < 0:
        return -1
    return _run_expif_complete(
        model,
        v0,
        refractory_remaining,
        n_steps,
        current,
        voltage_addr,
        refractory_addr,
        event_addr,
        True,
    )
