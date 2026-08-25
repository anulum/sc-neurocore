# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC stochastic rate-adaptation test support

"""Independent RK4 reference for the retained SC adaptation recurrence."""

from __future__ import annotations

import math

from sc_neurocore.neurons.models.sc_stochastic_rate_adaptation import (
    SCStochasticRateAdaptationNeuron,
)


def rk4_reference(neuron: SCStochasticRateAdaptationNeuron, current: float) -> tuple[float, float]:
    """Return an independent adaptation and hazard candidate."""

    def rhs(adaptation: float) -> tuple[float, float]:
        rate = neuron._f_onset(current - adaptation)
        return -adaptation / neuron.tau_a + neuron.delta_a * rate, rate

    k1, r1 = rhs(neuron.a)
    k2, r2 = rhs(neuron.a + 0.5 * neuron.dt * k1)
    k3, r3 = rhs(neuron.a + 0.5 * neuron.dt * k2)
    k4, r4 = rhs(neuron.a + neuron.dt * k3)
    next_a = neuron.a + neuron.dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    average_rate = (r1 + 2.0 * r2 + 2.0 * r3 + r4) / 6.0
    probability = -math.expm1(-average_rate * neuron.dt / 1000.0)
    return next_a, probability
