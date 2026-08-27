# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Larter-Breakspear cortical neural mass

from std.math import isfinite, tanh

comptime State = SIMD[DType.float64, 4]

struct LarterBreakspear(Copyable, Movable):
    var v: Float64
    var w: Float64
    var z: Float64
    var g_ca: Float64
    var g_na: Float64
    var g_k: Float64
    var g_l: Float64
    var v_ca: Float64
    var v_na: Float64
    var v_k: Float64
    var v_l: Float64
    var t_ca: Float64
    var t_na: Float64
    var t_k: Float64
    var delta_ca: Float64
    var delta_na: Float64
    var delta_k: Float64
    var phi: Float64
    var tau_k: Float64
    var b: Float64
    var a_ee: Float64
    var a_ei: Float64
    var a_ie: Float64
    var a_ne: Float64
    var a_ni: Float64
    var r_nmda: Float64
    var coupling_balance: Float64
    var v_t: Float64
    var z_t: Float64
    var delta_v: Float64
    var delta_z: Float64
    var q_v_max: Float64
    var q_z_max: Float64
    var i_ext: Float64
    var t_scale: Float64
    var dt: Float64

    def __init__(out self):
        self.v, self.w, self.z = 0.1, 0.1, 0.1
        self.g_ca, self.g_na, self.g_k, self.g_l = 1.1, 6.7, 2.0, 0.5
        self.v_ca, self.v_na, self.v_k, self.v_l = 1.0, 0.53, -0.7, -0.5
        self.t_ca, self.t_na, self.t_k = -0.01, 0.3, 0.0
        self.delta_ca, self.delta_na, self.delta_k = 0.15, 0.15, 0.3
        self.phi, self.tau_k, self.b = 0.7, 1.0, 0.1
        self.a_ee, self.a_ei, self.a_ie = 0.4, 2.0, 2.0
        self.a_ne, self.a_ni, self.r_nmda = 1.0, 0.4, 0.25
        self.coupling_balance = 0.1
        self.v_t, self.z_t, self.delta_v, self.delta_z = 0.0, 0.0, 0.65, 0.7
        self.q_v_max, self.q_z_max = 1.0, 1.0
        self.i_ext, self.t_scale, self.dt = 0.3, 1.0, 0.01

    def sigmoid(self, value: Float64, threshold: Float64, width: Float64) -> Float64:
        return 0.5 * (1.0 + tanh((value - threshold) / width))

    def derivatives(self, state: State, coupling: Float64) -> State:
        var v, w, z = state[0], state[1], state[2]
        var m_ca = self.sigmoid(v, self.t_ca, self.delta_ca)
        var m_na = self.sigmoid(v, self.t_na, self.delta_na)
        var m_k = self.sigmoid(v, self.t_k, self.delta_k)
        var q_v = self.q_v_max * self.sigmoid(v, self.v_t, self.delta_v)
        var q_z = self.q_z_max * self.sigmoid(z, self.z_t, self.delta_z)
        var excitation = self.a_ee * ((1.0 - self.coupling_balance) * q_v + self.coupling_balance * coupling)
        var result = State(0.0)
        result[0] = self.t_scale * (-(self.g_ca + self.r_nmda*excitation)*m_ca*(v-self.v_ca) - self.g_k*w*(v-self.v_k) - self.g_l*(v-self.v_l) - (self.g_na*m_na+excitation)*(v-self.v_na) - self.a_ie*z*q_z + self.a_ne*self.i_ext)
        result[1] = self.t_scale * self.phi * (m_k-w) / self.tau_k
        result[2] = self.t_scale * self.b * (self.a_ni*self.i_ext + self.a_ei*v*q_v)
        return result

    def step(mut self, coupling: Float64) raises -> Float64:
        if not isfinite(coupling) or not isfinite(self.v) or not isfinite(self.w) or not isfinite(self.z) or self.w < 0.0 or self.w > 1.0 or self.dt <= 0.0:
            raise Error("invalid Larter-Breakspear state or input")
        var state = State(0.0)
        state[0], state[1], state[2] = self.v, self.w, self.z
        var k1 = self.derivatives(state, coupling)
        var k2 = self.derivatives(state + 0.5*self.dt*k1, coupling)
        var k3 = self.derivatives(state + 0.5*self.dt*k2, coupling)
        var k4 = self.derivatives(state + self.dt*k3, coupling)
        var candidate = state + self.dt*(k1+2.0*k2+2.0*k3+k4)/6.0
        for index in range(3):
            if not isfinite(candidate[index]):
                raise Error("Larter-Breakspear RK4 candidate must be finite")
        if candidate[1] < 0.0 or candidate[1] > 1.0:
            raise Error("Larter-Breakspear potassium gate left its domain")
        self.v, self.w, self.z = candidate[0], candidate[1], candidate[2]
        return self.v

    def simulate(mut self, n_steps: Int, coupling: Float64) raises -> Float64:
        for _ in range(n_steps):
            _ = self.step(coupling)
        return self.v

def main() raises:
    var anchor = LarterBreakspear()
    _ = anchor.step(0.0)
    print(anchor.v, anchor.w, anchor.z)
    var trajectory = LarterBreakspear()
    _ = trajectory.simulate(10000, 0.0)
    print(trajectory.v, trajectory.w, trajectory.z)
