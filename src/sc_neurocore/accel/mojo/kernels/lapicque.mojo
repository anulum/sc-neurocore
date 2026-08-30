# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Lapicque exact-flow simulator
#
# Build:
#   mojo build --emit shared-lib -o liblapicque.so lapicque.mojo
#
# The caller supplies n_steps+1 Float64 slots: the post-step voltage trace and
# final voltage. Rejected contracts leave the buffer untouched because a
# validation pass completes before emission.

from std.math import exp, isfinite
from std.memory import UnsafePointer


struct Lapicque(Copyable, Movable):
    var v_rest: Float64
    var v_reset: Float64
    var v_threshold: Float64
    var tau: Float64
    var resistance: Float64
    var dt: Float64

    def __init__(
        out self,
        v_rest: Float64,
        v_reset: Float64,
        v_threshold: Float64,
        tau: Float64,
        resistance: Float64,
        dt: Float64,
    ):
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.tau = tau
        self.resistance = resistance
        self.dt = dt

    @always_inline
    def valid(self, v: Float64) -> Bool:
        return (
            isfinite(v)
            and isfinite(self.v_rest)
            and isfinite(self.v_reset)
            and isfinite(self.v_threshold)
            and self.v_threshold > self.v_rest
            and self.v_threshold > self.v_reset
            and v < self.v_threshold
            and isfinite(self.tau)
            and self.tau > 0.0
            and isfinite(self.resistance)
            and self.resistance > 0.0
            and isfinite(self.dt)
            and self.dt > 0.0
        )


@always_inline
def _candidate(model: Lapicque, v: Float64, current: Float64) -> Float64:
    var v_inf = model.v_rest + model.resistance * current
    var decay = exp(-model.dt / model.tau)
    return v_inf + (v - v_inf) * decay


def lapicque_valid(
    v: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau: Float64,
    resistance: Float64,
    dt: Float64,
) -> Bool:
    return Lapicque(v_rest, v_reset, v_threshold, tau, resistance, dt).valid(v)


def lapicque_step_spike(
    v: Float64,
    current: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau: Float64,
    resistance: Float64,
    dt: Float64,
) -> Int:
    var model = Lapicque(v_rest, v_reset, v_threshold, tau, resistance, dt)
    if not isfinite(current) or not model.valid(v):
        return 0
    var next_v = _candidate(model, v, current)
    if not isfinite(next_v):
        return 0
    return Int(next_v >= v_threshold)


def _run_lapicque(
    model: Lapicque,
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
def lapicque_simulate_c(
    v0: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau: Float64,
    resistance: Float64,
    dt: Float64,
    n_steps: Int,
    current: Float64,
    output_addr: Int,
) -> Int64:
    if n_steps < 0 or output_addr == 0 or not isfinite(current):
        return -1
    var model = Lapicque(v_rest, v_reset, v_threshold, tau, resistance, dt)
    var validated = _run_lapicque(
        model, v0, n_steps, current, output_addr, False
    )
    if validated < 0:
        return -1
    return _run_lapicque(model, v0, n_steps, current, output_addr, True)


def _run_lapicque_complete(
    v0: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau: Float64,
    resistance: Float64,
    dt: Float64,
    capacitance: Float64,
    series_resistance: Float64,
    polarization_resistance: Float64,
    excited0: Bool,
    source_profile: Bool,
    n_steps: Int,
    drive: Float64,
    voltage_addr: Int,
    event_addr: Int,
    write_output: Bool,
) -> Int64:
    var v = v0
    var excited = excited0
    var common_valid = (
        isfinite(v)
        and isfinite(v_threshold)
        and v_threshold > 0.0
        and isfinite(dt)
        and dt > 0.0
    )
    var profile_valid: Bool
    if source_profile:
        profile_valid = (
            (excited or v < v_threshold)
            and isfinite(capacitance)
            and capacitance > 0.0
            and isfinite(series_resistance)
            and series_resistance > 0.0
            and isfinite(polarization_resistance)
            and polarization_resistance > 0.0
        )
    else:
        profile_valid = (
            not excited
            and isfinite(v_rest)
            and isfinite(v_reset)
            and v_threshold > v_rest
            and v_threshold > v_reset
            and v < v_threshold
            and isfinite(tau)
            and tau > 0.0
            and isfinite(resistance)
            and resistance > 0.0
        )
    if not common_valid or not profile_valid or not isfinite(drive):
        return -1

    var voltage = UnsafePointer[Float64, MutAnyOrigin](
        unsafe_from_address=voltage_addr
    )
    var events = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=event_addr)
    var event_count: Int64 = 0
    for index in range(n_steps):
        var v_inf: Float64
        var decay: Float64
        if source_profile:
            var total_resistance = series_resistance + polarization_resistance
            var beta = (
                capacitance
                * series_resistance
                * polarization_resistance
                / total_resistance
            )
            v_inf = drive * polarization_resistance / total_resistance
            decay = exp(-dt / beta)
        else:
            v_inf = v_rest + resistance * drive
            decay = exp(-dt / tau)
        var next_v = v_inf + (v - v_inf) * decay
        if not isfinite(v_inf) or not isfinite(decay) or not isfinite(next_v):
            return -1
        var event: UInt8 = 0
        if source_profile:
            if not excited and next_v >= v_threshold:
                excited = True
                event = 1
                event_count += 1
            v = next_v
        elif next_v >= v_threshold:
            v = v_reset
            event = 1
            event_count += 1
        else:
            v = next_v
        if write_output:
            voltage[index] = v
            events[index] = event
    if write_output:
        voltage[n_steps] = v
    return event_count


@export
def lapicque_simulate_complete_c(
    v0: Float64,
    v_rest: Float64,
    v_reset: Float64,
    v_threshold: Float64,
    tau: Float64,
    resistance: Float64,
    dt: Float64,
    capacitance: Float64,
    series_resistance: Float64,
    polarization_resistance: Float64,
    excited: Int,
    source_profile: Int,
    n_steps: Int,
    drive: Float64,
    voltage_addr: Int,
    event_addr: Int,
) -> Int64:
    if (
        n_steps < 0
        or voltage_addr == 0
        or event_addr == 0
        or (excited != 0 and excited != 1)
        or (source_profile != 0 and source_profile != 1)
    ):
        return -1
    var validated = _run_lapicque_complete(
        v0,
        v_rest,
        v_reset,
        v_threshold,
        tau,
        resistance,
        dt,
        capacitance,
        series_resistance,
        polarization_resistance,
        excited != 0,
        source_profile != 0,
        n_steps,
        drive,
        voltage_addr,
        event_addr,
        False,
    )
    if validated < 0:
        return -1
    return _run_lapicque_complete(
        v0,
        v_rest,
        v_reset,
        v_threshold,
        tau,
        resistance,
        dt,
        capacitance,
        series_resistance,
        polarization_resistance,
        excited != 0,
        source_profile != 0,
        n_steps,
        drive,
        voltage_addr,
        event_addr,
        True,
    )
