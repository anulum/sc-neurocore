# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Medvedev map reference calculations

"""Independent reference calculations shared by Medvedev map tests."""

from __future__ import annotations

import math

from sc_neurocore.neurons.models.medvedev_map import MedvedevMapNeuron


def _boundaries(neuron: MedvedevMapNeuron) -> tuple[float, float, float]:
    """Return independently derived source branch boundaries."""
    return (
        neuron.beta_0 / (neuron.delta - neuron.beta_0),
        neuron.beta_hc / (neuron.delta - neuron.beta_hc),
        neuron.beta_sn / (neuron.delta - neuron.beta_sn),
    )


def _inner_reference(neuron: MedvedevMapNeuron, state: float, current: float) -> float:
    """Independently evaluate the calibrated Eqs. 4.8 and 4.13 branch."""
    u_1 = (1.0 - neuron.alpha_t0) * state + neuron.alpha_t0 * neuron.f_0
    gap = neuron.beta_hc - neuron.delta * u_1 / (1.0 + u_1)
    inner = neuron.f_1
    if gap > 0.0:
        scale = math.exp(neuron.homoclinic_exponent * math.log(neuron.d * gap))
        inner = scale * (u_1 - neuron.f_1) + neuron.f_1
    return inner + neuron.input_gain * current
