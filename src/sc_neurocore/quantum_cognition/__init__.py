# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.quantum_cognition -- Tier: experimental

"""Quantum cognition layer: Fisher-Posner hypothesis.

Experimental module coupling classical Leaky Integrate-and-Fire (LIF)
dynamics with an exact small-spin-state Posner layer.  The core idea is
that :sup:`31`\\ P nuclear spins in Posner calcium phosphate molecules
(Ca₉(PO₄)₆) may preserve entanglement long enough to modulate ATP
hydrolysis probability, thereby providing a quantum-metabolic coupling
dimension absent from classical neuron models.

**Theoretical basis:** Fisher, M. P. A. "Quantum cognition: The possibility
of processing with nuclear spins in the brain." *Annals of Physics* 362,
593–602 (2015).  doi:10.1016/j.aop.2015.08.020

.. warning::
    This module implements speculative neuroscience.  The Fisher-Posner
    hypothesis is not experimentally confirmed.  Results should not be
    interpreted as validated quantum biology.  Use for exploratory
    research only.

Install extras::

    pip install sc-neurocore[quantum-cognition]
"""

__tier__ = "experimental"

from .bridge_adapter import FisherPosnerQuantumBridge, compute_max_qubits
from .content_indexer import ContentChunk, embed_chunks, index_gotm_repo
from .dashboard import TerminalDashboard
from .fisher_posner import HybridFisherPosnerLIF, HybridFisherPosnerLIFNeuron
from .fs_watcher import GOTMWatcher
from .gotm_brain import GOTMBrain
from .kane_mapper import KaneRegisterLayout, KaneSiliconMapper
from .radical_pair import RadicalPairModel, RadicalPairParams
from .spin_pool import SpinPoolMPS
from .studio_hook import QuantumStudioHook

__all__ = [
    "SpinPoolMPS",
    "HybridFisherPosnerLIF",
    "HybridFisherPosnerLIFNeuron",
    "FisherPosnerQuantumBridge",
    "compute_max_qubits",
    "QuantumStudioHook",
    "ContentChunk",
    "index_gotm_repo",
    "embed_chunks",
    "GOTMBrain",
    "RadicalPairModel",
    "RadicalPairParams",
    "KaneSiliconMapper",
    "KaneRegisterLayout",
    "TerminalDashboard",
    "GOTMWatcher",
]
