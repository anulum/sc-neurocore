# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Connor-Stevens conductance-model simulator
#
# Build:
#   mojo build --emit shared-lib -o libconnor_stevens.so connor_stevens.mojo
#
# The C ABI advances the six-state Connor-Stevens model with the maintained
# candidate-first RK4 macro-step: Int(1 / max(dt, 0.001)) sub-steps, followed by
# an upward v-threshold crossing observation. The caller supplies n_steps + 6
# Float64 slots: the voltage trace followed by final (v, m, h, n, a, b).
# A negative return rejects invalid input or a non-finite/out-of-envelope
# candidate; callers must not commit the output state in that case.
#
# Reference: Connor, J.A. & Stevens, C.F. (1971). J. Physiol. 213:31-53.

from std.math import exp, isfinite, log
from std.memory import UnsafePointer


comptime State = SIMD[DType.float64, 8]


@always_inline
def _finite_state(state: State) -> Bool:
    return (
        isfinite(state[0])
        and isfinite(state[1])
        and isfinite(state[2])
        and isfinite(state[3])
        and isfinite(state[4])
        and isfinite(state[5])
    )


@always_inline
def _candidate_valid(state: State) -> Bool:
    return (
        _finite_state(state)
        and state[0] >= -250.0
        and state[0] <= 250.0
        and state[1] >= -0.05
        and state[1] <= 1.05
        and state[2] >= -0.05
        and state[2] <= 1.05
        and state[3] >= -0.05
        and state[3] <= 1.05
        and state[4] >= -0.05
        and state[4] <= 1.5
        and state[5] >= -0.05
        and state[5] <= 1.05
    )


@always_inline
def _supported_exp_argument(value: Float64) -> Bool:
    return isfinite(value) and value <= 700.0


struct ConnorStevens(Copyable, Movable):
    var g_na: Float64
    var g_k: Float64
    var g_a: Float64
    var g_l: Float64
    var e_na: Float64
    var e_k: Float64
    var e_a: Float64
    var e_l: Float64
    var c_m: Float64
    var dt: Float64
    var v_threshold: Float64

    def __init__(
        out self,
        g_na: Float64,
        g_k: Float64,
        g_a: Float64,
        g_l: Float64,
        e_na: Float64,
        e_k: Float64,
        e_a: Float64,
        e_l: Float64,
        c_m: Float64,
        dt: Float64,
        v_threshold: Float64,
    ):
        self.g_na = g_na
        self.g_k = g_k
        self.g_a = g_a
        self.g_l = g_l
        self.e_na = e_na
        self.e_k = e_k
        self.e_a = e_a
        self.e_l = e_l
        self.c_m = c_m
        self.dt = dt
        self.v_threshold = v_threshold

    @always_inline
    def parameters_valid(self) -> Bool:
        return (
            isfinite(self.g_na)
            and isfinite(self.g_k)
            and isfinite(self.g_a)
            and isfinite(self.g_l)
            and isfinite(self.e_na)
            and isfinite(self.e_k)
            and isfinite(self.e_a)
            and isfinite(self.e_l)
            and isfinite(self.c_m)
            and isfinite(self.dt)
            and isfinite(self.v_threshold)
            and self.g_na >= 0.0
            and self.g_k >= 0.0
            and self.g_a >= 0.0
            and self.g_l >= 0.0
            and self.c_m > 0.0
            and self.dt > 0.0
        )

    @always_inline
    def rates_valid(self, v: Float64) -> Bool:
        if not isfinite(v):
            return False
        var alpha_m_arg = -(v + 29.7) / 10.0
        var beta_m_arg = -(v + 54.7) / 18.0
        var alpha_h_arg = -(v + 48.0) / 20.0
        var beta_h_arg = -(v + 18.0) / 10.0
        var alpha_n_arg = -(v + 45.7) / 10.0
        var beta_n_arg = -(v + 55.7) / 80.0
        var a_num_arg = (v + 94.22) / 31.84
        var a_den_arg = (v + 1.17) / 28.93
        var tau_a_arg = (v + 55.96) / 20.12
        var b_inf_arg = (v + 53.3) / 14.54
        var tau_b_arg = (v + 50.0) / 16.027
        if not (
            _supported_exp_argument(alpha_m_arg)
            and _supported_exp_argument(beta_m_arg)
            and _supported_exp_argument(alpha_h_arg)
            and _supported_exp_argument(beta_h_arg)
            and _supported_exp_argument(alpha_n_arg)
            and _supported_exp_argument(beta_n_arg)
            and _supported_exp_argument(a_num_arg)
            and _supported_exp_argument(a_den_arg)
            and _supported_exp_argument(tau_a_arg)
            and _supported_exp_argument(b_inf_arg)
            and _supported_exp_argument(tau_b_arg)
        ):
            return False
        var a_base = 0.0761 * exp(a_num_arg) / (1.0 + exp(a_den_arg))
        return isfinite(a_base) and a_base > 0.0

    @always_inline
    def derivatives(self, state: State, current: Float64) -> State:
        var v = state[0]
        var m = state[1]
        var h = state[2]
        var n = state[3]
        var a = state[4]
        var b = state[5]

        var delta_m = v + 29.7
        var x_m = delta_m / 10.0
        var alpha_m: Float64
        if abs(x_m) < 1.0e-9:
            alpha_m = 3.8
        else:
            alpha_m = 0.38 * delta_m / (1.0 - exp(-x_m))
        var beta_m = 15.2 * exp(-(v + 54.7) / 18.0)
        var alpha_h = 0.266 * exp(-(v + 48.0) / 20.0)
        var beta_h = 3.8 / (1.0 + exp(-(v + 18.0) / 10.0))
        var delta_n = v + 45.7
        var x_n = delta_n / 10.0
        var alpha_n: Float64
        if abs(x_n) < 1.0e-9:
            alpha_n = 0.2
        else:
            alpha_n = 0.02 * delta_n / (1.0 - exp(-x_n))
        var beta_n = 0.25 * exp(-(v + 55.7) / 80.0)
        var a_base = 0.0761 * exp((v + 94.22) / 31.84) / (1.0 + exp((v + 1.17) / 28.93))
        var a_inf = exp(log(a_base) / 3.0)
        var tau_a = 0.3632 + 1.158 / (1.0 + exp((v + 55.96) / 20.12))
        var b_base = 1.0 / (1.0 + exp((v + 53.3) / 14.54))
        var b_squared = b_base * b_base
        var b_inf = b_squared * b_squared
        var tau_b = 1.24 + 2.678 / (1.0 + exp((v + 50.0) / 16.027))

        var m_squared = m * m
        var n_squared = n * n
        var a_squared = a * a
        var i_na = self.g_na * m_squared * m * h * (v - self.e_na)
        var i_k = self.g_k * n_squared * n_squared * (v - self.e_k)
        var i_a = self.g_a * a_squared * a * b * (v - self.e_a)
        var i_l = self.g_l * (v - self.e_l)
        var result = State(0.0)
        result[0] = (-i_na - i_k - i_a - i_l + current) / self.c_m
        result[1] = alpha_m * (1.0 - m) - beta_m * m
        result[2] = alpha_h * (1.0 - h) - beta_h * h
        result[3] = alpha_n * (1.0 - n) - beta_n * n
        result[4] = (a_inf - a) / tau_a
        result[5] = (b_inf - b) / tau_b
        return result


@export
def connor_stevens_simulate_c(
    v0: Float64,
    m0: Float64,
    h0: Float64,
    n0: Float64,
    a0: Float64,
    b0: Float64,
    g_na: Float64,
    g_k: Float64,
    g_a: Float64,
    g_l: Float64,
    e_na: Float64,
    e_k: Float64,
    e_a: Float64,
    e_l: Float64,
    c_m: Float64,
    dt: Float64,
    v_threshold: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    if n_steps < 0 or trace_addr == 0 or not isfinite(current):
        return -1
    var neuron = ConnorStevens(
        g_na, g_k, g_a, g_l, e_na, e_k, e_a, e_l, c_m, dt, v_threshold
    )
    var state = State(0.0)
    state[0] = v0
    state[1] = m0
    state[2] = h0
    state[3] = n0
    state[4] = a0
    state[5] = b0
    if not neuron.parameters_valid() or not _candidate_valid(state):
        return -1

    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var substep_floor = neuron.dt
    if substep_floor < 0.001:
        substep_floor = 0.001
    var substeps = Int(1.0 / substep_floor)
    var spikes: Int64 = 0
    for step in range(n_steps):
        var v_previous = state[0]
        for _ in range(substeps):
            if not neuron.rates_valid(state[0]):
                return -1
            var k1 = neuron.derivatives(state, current)
            if not _finite_state(k1):
                return -1
            var k2_state = state + 0.5 * neuron.dt * k1
            if not _finite_state(k2_state) or not neuron.rates_valid(k2_state[0]):
                return -1
            var k2 = neuron.derivatives(k2_state, current)
            if not _finite_state(k2):
                return -1
            var k3_state = state + 0.5 * neuron.dt * k2
            if not _finite_state(k3_state) or not neuron.rates_valid(k3_state[0]):
                return -1
            var k3 = neuron.derivatives(k3_state, current)
            if not _finite_state(k3):
                return -1
            var k4_state = state + neuron.dt * k3
            if not _finite_state(k4_state) or not neuron.rates_valid(k4_state[0]):
                return -1
            var k4 = neuron.derivatives(k4_state, current)
            if not _finite_state(k4):
                return -1
            state = state + neuron.dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            if not _candidate_valid(state):
                return -1
        trace[step] = state[0]
        if state[0] >= neuron.v_threshold and v_previous < neuron.v_threshold:
            spikes += 1

    trace[n_steps] = state[0]
    trace[n_steps + 1] = state[1]
    trace[n_steps + 2] = state[2]
    trace[n_steps + 3] = state[3]
    trace[n_steps + 4] = state[4]
    trace[n_steps + 5] = state[5]
    return spikes
