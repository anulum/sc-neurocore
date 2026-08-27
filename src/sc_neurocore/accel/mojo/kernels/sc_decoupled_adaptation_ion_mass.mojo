# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained three-state project ion-mass recurrence

from std.math import isfinite, tanh

comptime State = SIMD[DType.float64, 4]

struct SCDecoupledAdaptationIonMass(Copyable, Movable):
    var v: Float64
    var w: Float64
    var z: Float64
    var dt: Float64

    def __init__(out self):
        self.v, self.w, self.z, self.dt = -0.5, 0.0, 0.0, 0.01

    def gate(self, v: Float64, midpoint: Float64, width: Float64) -> Float64:
        return 0.5 * (1.0 + tanh((v-midpoint)/width))

    def derivatives(self, state: State, coupling: Float64) -> State:
        var v, w, z = state[0], state[1], state[2]
        var result = State(0.0)
        result[0] = -1.1*self.gate(v,-0.01,0.15)*(v-1.0) - 6.7*self.gate(v,0.12,0.15)*(v-0.53) - 2.0*w*(v+0.7) - 0.5*(v+0.5) + 0.3 + coupling + 0.36*v
        result[1] = 0.7*(self.gate(v,0.0,0.3)-w)
        result[2] = 0.1*(v+0.5-z)
        return result

    def step(mut self, coupling: Float64) raises -> Float64:
        if not isfinite(coupling) or not isfinite(self.v) or not isfinite(self.w) or not isfinite(self.z):
            raise Error("invalid SC ion-mass state or input")
        var state = State(0.0)
        state[0], state[1], state[2] = self.v, self.w, self.z
        var k1 = self.derivatives(state, coupling)
        var k2 = self.derivatives(state + 0.5*self.dt*k1, coupling)
        var k3 = self.derivatives(state + 0.5*self.dt*k2, coupling)
        var k4 = self.derivatives(state + self.dt*k3, coupling)
        var candidate = state + self.dt*(k1+2.0*k2+2.0*k3+k4)/6.0
        for index in range(3):
            if not isfinite(candidate[index]):
                raise Error("SC ion-mass candidate must be finite")
        if candidate[1] < 0.0 or candidate[1] > 1.0:
            raise Error("SC ion-mass potassium gate left its domain")
        self.v, self.w, self.z = candidate[0], candidate[1], candidate[2]
        return self.v

def main() raises:
    var neuron = SCDecoupledAdaptationIonMass()
    _ = neuron.step(0.0)
    print(neuron.v, neuron.w, neuron.z)
