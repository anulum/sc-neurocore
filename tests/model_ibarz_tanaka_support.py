# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared Ibarz-Tanaka source-map test support

"""Independent source recurrence for Ibarz-Tanaka map tests."""

from __future__ import annotations

from sc_neurocore.neurons.models.ibarz_tanaka_map import IbarzTanakaMapNeuron


def _reference_step(neuron: IbarzTanakaMapNeuron, current: float) -> tuple[float, float, int]:
    """Independently evaluate Ibarz et al. (2007), Eqs. 2-3."""
    lower = -1.0 - neuron.alpha / 2.0
    upper = 1.0 + current + neuron.u
    if neuron.v < lower:
        v_next = -(neuron.alpha**2) / 4.0 - neuron.alpha + current + neuron.u
    elif neuron.v <= 0.0:
        v_next = neuron.alpha * neuron.v + (neuron.v + 1.0) ** 2 + current + neuron.u
    elif neuron.v < upper:
        v_next = upper
    else:
        v_next = -1.0
    u_next = neuron.u - neuron.mu * (neuron.v + 1.0 - neuron.sigma)
    return v_next, u_next, int(neuron.v >= upper)
