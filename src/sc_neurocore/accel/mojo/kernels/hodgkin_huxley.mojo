# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Hodgkin-Huxley conductance-model simulator
#
# Build:
#   mojo build --emit shared-lib -o libhodgkin_huxley.so hodgkin_huxley.mojo
#
# The C ABI mirrors the maintained baseline-Euler macro-step: round(1 / dt)
# explicit substeps, each updating m/h/n first and then voltage with the new
# gates. The caller supplies n_steps + 4 Float64 slots: voltage trace followed
# by final (v, m, h, n). A negative return rejects invalid input or an invalid
# candidate; callers must not commit the output state in that case.
#
# Reference: Hodgkin, A.L. & Huxley, A.F. (1952). J. Physiol. 117:500-544.

from std.math import exp, floor, isfinite
from std.memory import UnsafePointer


comptime State = SIMD[DType.float64, 4]


@always_inline
def _candidate_valid(state: State) -> Bool:
    return (
        isfinite(state[0])
        and isfinite(state[1])
        and isfinite(state[2])
        and isfinite(state[3])
        and state[0] >= -250.0
        and state[0] <= 250.0
        and state[1] >= -0.05
        and state[1] <= 1.05
        and state[2] >= -0.05
        and state[2] <= 1.05
        and state[3] >= -0.05
        and state[3] <= 1.05
    )


@always_inline
def _supported_exp_argument(value: Float64) -> Bool:
    return isfinite(value) and value <= 700.0


@always_inline
def _round_half_even_nonnegative(value: Float64) -> Int:
    var lower = Int(floor(value))
    var fraction = value - Float64(lower)
    if fraction > 0.5:
        return lower + 1
    if fraction < 0.5 or lower % 2 == 0:
        return lower
    return lower + 1


struct HodgkinHuxley(Copyable, Movable):
    var c_m: Float64
    var g_na: Float64
    var g_k: Float64
    var g_l: Float64
    var e_na: Float64
    var e_k: Float64
    var e_l: Float64
    var dt: Float64
    var v_threshold: Float64

    def __init__(
        out self,
        c_m: Float64,
        g_na: Float64,
        g_k: Float64,
        g_l: Float64,
        e_na: Float64,
        e_k: Float64,
        e_l: Float64,
        dt: Float64,
        v_threshold: Float64,
    ):
        self.c_m = c_m
        self.g_na = g_na
        self.g_k = g_k
        self.g_l = g_l
        self.e_na = e_na
        self.e_k = e_k
        self.e_l = e_l
        self.dt = dt
        self.v_threshold = v_threshold

    @always_inline
    def parameters_valid(self) -> Bool:
        return (
            isfinite(self.c_m)
            and isfinite(self.g_na)
            and isfinite(self.g_k)
            and isfinite(self.g_l)
            and isfinite(self.e_na)
            and isfinite(self.e_k)
            and isfinite(self.e_l)
            and isfinite(self.dt)
            and isfinite(self.v_threshold)
            and self.c_m > 0.0
            and self.g_na >= 0.0
            and self.g_k >= 0.0
            and self.g_l >= 0.0
            and self.dt > 0.0
        )

    @always_inline
    def rates_valid(self, v: Float64) -> Bool:
        return (
            isfinite(v)
            and _supported_exp_argument(-(v + 40.0) / 10.0)
            and _supported_exp_argument(-(v + 65.0) / 18.0)
            and _supported_exp_argument(-(v + 65.0) / 20.0)
            and _supported_exp_argument(-(v + 35.0) / 10.0)
            and _supported_exp_argument(-(v + 55.0) / 10.0)
            and _supported_exp_argument(-(v + 65.0) / 80.0)
        )

    @always_inline
    def alpha_m(self, v: Float64) -> Float64:
        var delta = v + 40.0
        if abs(delta) < 1.0e-7:
            return 1.0
        return 0.1 * delta / (1.0 - exp(-delta / 10.0))

    @always_inline
    def beta_m(self, v: Float64) -> Float64:
        return 4.0 * exp(-(v + 65.0) / 18.0)

    @always_inline
    def alpha_h(self, v: Float64) -> Float64:
        return 0.07 * exp(-(v + 65.0) / 20.0)

    @always_inline
    def beta_h(self, v: Float64) -> Float64:
        return 1.0 / (1.0 + exp(-(v + 35.0) / 10.0))

    @always_inline
    def alpha_n(self, v: Float64) -> Float64:
        var delta = v + 55.0
        if abs(delta) < 1.0e-7:
            return 0.1
        return 0.01 * delta / (1.0 - exp(-delta / 10.0))

    @always_inline
    def beta_n(self, v: Float64) -> Float64:
        return 0.125 * exp(-(v + 65.0) / 80.0)


@export
def hodgkin_huxley_simulate_c(
    v0: Float64,
    m0: Float64,
    h0: Float64,
    n0: Float64,
    c_m: Float64,
    g_na: Float64,
    g_k: Float64,
    g_l: Float64,
    e_na: Float64,
    e_k: Float64,
    e_l: Float64,
    dt: Float64,
    v_threshold: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    if n_steps < 0 or trace_addr == 0 or not isfinite(current):
        return -1
    var neuron = HodgkinHuxley(c_m, g_na, g_k, g_l, e_na, e_k, e_l, dt, v_threshold)
    var state = State(0.0)
    state[0] = v0
    state[1] = m0
    state[2] = h0
    state[3] = n0
    if not neuron.parameters_valid() or not _candidate_valid(state):
        return -1

    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var substeps = _round_half_even_nonnegative(1.0 / neuron.dt)
    var spikes: Int64 = 0
    for step in range(n_steps):
        var v_previous = state[0]
        for _ in range(substeps):
            var v = state[0]
            if not neuron.rates_valid(v):
                return -1
            var am = neuron.alpha_m(v)
            var bm = neuron.beta_m(v)
            var ah = neuron.alpha_h(v)
            var bh = neuron.beta_h(v)
            var an = neuron.alpha_n(v)
            var bn = neuron.beta_n(v)
            if not (
                isfinite(am)
                and isfinite(bm)
                and isfinite(ah)
                and isfinite(bh)
                and isfinite(an)
                and isfinite(bn)
            ):
                return -1
            state[1] += (am * (1.0 - state[1]) - bm * state[1]) * neuron.dt
            state[2] += (ah * (1.0 - state[2]) - bh * state[2]) * neuron.dt
            state[3] += (an * (1.0 - state[3]) - bn * state[3]) * neuron.dt
            var m_squared = state[1] * state[1]
            var n_squared = state[3] * state[3]
            var i_na = neuron.g_na * m_squared * state[1] * state[2] * (v - neuron.e_na)
            var i_k = neuron.g_k * n_squared * n_squared * (v - neuron.e_k)
            var i_l = neuron.g_l * (v - neuron.e_l)
            state[0] += (-i_na - i_k - i_l + current) / neuron.c_m * neuron.dt
            if not _candidate_valid(state):
                return -1
        trace[step] = state[0]
        if state[0] >= neuron.v_threshold and v_previous < neuron.v_threshold:
            spikes += 1

    trace[n_steps] = state[0]
    trace[n_steps + 1] = state[1]
    trace[n_steps + 2] = state[2]
    trace[n_steps + 3] = state[3]
    return spikes
