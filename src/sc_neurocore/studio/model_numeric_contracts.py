# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The fixed-point formats Studio offers a model

"""The fixed-point formats Studio compiles a catalogue model at.

Studio offers :data:`STUDIO_Q_FORMATS`, the word geometries every Studio stage
handles: the RTL compile, the generated bit-true C kernel (words up to 32 bits)
and the co-simulation harness. A format is offered for a model only when the
model's neuron is representable in it, as
:func:`~sc_neurocore.compiler.hardware_numeric_contract.hardware_numeric_contract`
decides; the compile refuses any other format.
"""

from __future__ import annotations

from sc_neurocore.compiler.hardware_numeric_contract import (
    HardwareNumericContract,
    hardware_numeric_contract,
)
from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.neurons.universal_dsl import UniversalNeuron

STUDIO_Q_FORMATS: tuple[str, ...] = ("Q8.8", "Q16.16")
"""The candidate formats, smallest word first."""


def studio_numeric_contracts(neuron: UniversalNeuron) -> dict[str, HardwareNumericContract]:
    """Return the hardware numeric contract of ``neuron`` at every candidate format.

    Parameters
    ----------
    neuron:
        The instantiated schema neuron Studio would compile.

    Returns
    -------
    dict
        Contract per :data:`STUDIO_Q_FORMATS` label, in that order.
    """
    equation_neuron = neuron.to_equation_neuron()
    return {
        label: hardware_numeric_contract(equation_neuron, QFormat.from_string(label))
        for label in STUDIO_Q_FORMATS
    }


__all__ = ["STUDIO_Q_FORMATS", "studio_numeric_contracts"]
