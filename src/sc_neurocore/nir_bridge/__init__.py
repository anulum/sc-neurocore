# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR (Neuromorphic Intermediate Representation) bridge

"""
NIR integration for SC-NeuroCore.

Provides bidirectional conversion between NIR graphs and SC-NeuroCore
networks.

    >>> import nir
    >>> from sc_neurocore.nir_bridge import from_nir
    >>> graph = nir.read("model.nir")
    >>> network = from_nir(graph, dt=1.0)
    >>> network.run(inputs, steps=100)
"""

from .parser import from_nir
from .export import to_nir

__all__ = ["from_nir", "to_nir"]
