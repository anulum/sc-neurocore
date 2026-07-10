# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Catalogue re-export of the public StochasticLIFNeuron

"""Catalogue registration re-export for :class:`StochasticLIFNeuron`.

The implementation lives at :mod:`sc_neurocore.neurons.stochastic_lif` so the
package root can keep a torch-free, SC-first public import. This module exists
so the model appears in ``_CLASS_TO_MODULE``, the descriptor corpus, and Studio
catalogue readiness without duplicating dynamics.
"""

from __future__ import annotations

from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron

__all__ = ["StochasticLIFNeuron"]
