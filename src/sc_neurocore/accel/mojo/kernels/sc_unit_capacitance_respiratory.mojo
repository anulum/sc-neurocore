# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo retained unit-capacitance respiratory identity

from butera_respiratory import ButeraRespiratory


# Count-neutral wrapper preserving the former SC defaults and recurrence.
struct SCUnitCapacitanceRespiratory(Copyable, Movable):
    var inner: ButeraRespiratory

    def __init__(out self):
        self.inner = ButeraRespiratory()
        self.inner.capacitance = 1.0
        self.inner.e_syn = -10.0

    def step(mut self, current: Float64) raises -> Int:
        return self.inner.step(current)

    def simulate(mut self, n_steps: Int, current: Float64) raises -> Int:
        return self.inner.simulate(n_steps, current)


def main() raises:
    var anchor = SCUnitCapacitanceRespiratory()
    var event = anchor.step(12.5)
    print(anchor.inner.v, anchor.inner.n, anchor.inner.h_nap, event)
    var neuron = SCUnitCapacitanceRespiratory()
    var spikes = neuron.simulate(20000, 20.0)
    print(neuron.inner.v, neuron.inner.n, neuron.inner.h_nap, spikes)
    if spikes < 4 or spikes > 5:
        raise Error("SC unit-capacitance respiratory event envelope failed")
