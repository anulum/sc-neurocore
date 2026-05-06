# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — HybridFisherPosnerLIF model entry

"""Population-compatible Fisher-Posner quantum-metabolic LIF neuron.

Re-exports :class:`HybridFisherPosnerLIFNeuron` from the
``quantum_cognition`` subpackage so it can be used via the standard
``Population(model='HybridFisherPosnerLIFNeuron', n=N)`` dispatch.
"""

from sc_neurocore.quantum_cognition.fisher_posner import (
    HybridFisherPosnerLIFNeuron,
)

__all__ = ["HybridFisherPosnerLIFNeuron"]
