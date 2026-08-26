# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained unit-capacitance respiratory recurrence

from __future__ import annotations

from dataclasses import dataclass

from sc_neurocore.neurons.models.butera_respiratory import ButeraRespiratoryNeuron


@dataclass
class SCUnitCapacitanceRespiratoryNeuron(ButeraRespiratoryNeuron):
    """Retain the historical SC respiratory recurrence without paper attribution.

    The legacy implementation omitted the Butera Model 1 whole-cell
    capacitance divisor, which is equivalent to fixing capacitance to one in
    the repository's RK4 recurrence. This count-neutral identity preserves
    that established timing and event behavior while the literature-labelled
    class uses the source ``C = 21 pF`` equation.
    """

    capacitance: float = 1.0
    e_syn: float = -10.0
