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
# The caller supplies n_steps+2 Float64 slots: the post-step voltage trace,
# final voltage, and final refractory remainder. Rejected contracts leave the
# caller buffer untouched because a validation pass completes before emission.

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
    ):
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.v_rh = v_rh
        self.delta_t = delta_t
        self.tau = tau
        self.dt = dt
        self.refractory_period = refractory_period

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
