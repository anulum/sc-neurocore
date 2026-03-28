# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L3 Genomic-Epigenomic Layer (Stochastic Implementation)

from typing import Any, Optional

"""
SCPN L3: Genomic-Epigenomic Layer (Stochastic Implementation)
=============================================================

Implements Layer 3 of the SCPN framework: Genomic and epigenomic regulation
including CISS (Chiral-Induced Spin Selectivity), bioelectric patterns,
and chromatin dynamics.

Key Features:
- Stochastic gene expression dynamics
- CISS spin-filtering effects on DNA charge transport
- Chromatin state modulation via bitstreams
- Integration with bio/grn.py (Gene Regulatory Networks)

"""

from dataclasses import dataclass
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class L3_StochasticParameters:
    """Parameters for the Stochastic L3 Genomic Layer."""

    n_genes: int = 200
    n_regulatory_elements: int = 50
    bitstream_length: int = 1024

    # Gene expression dynamics
    transcription_rate: float = 0.1
    translation_rate: float = 0.2
    degradation_rate: float = 0.05

    # CISS (Chiral-Induced Spin Selectivity)
    ciss_efficiency: float = 0.8  # Spin polarization efficiency
    dna_chirality: float = 1.0  # Right-handed helix = +1

    # Chromatin dynamics
    methylation_rate: float = 0.01
    demethylation_rate: float = 0.02
    histone_mod_rate: float = 0.05

    # Bioelectric coupling
    bioelectric_coupling: float = 0.15
    membrane_potential_rest: float = -70.0  # mV

    # Inter-layer coupling
    neurochemical_coupling: float = 0.2  # From L2
    cellular_coupling: float = 0.1  # To L4


class L3_GenomicLayer:
    """
    Stochastic implementation of the Genomic-Epigenomic Layer.

    Models gene expression, epigenetic modifications, and bioelectric
    pattern formation using bitstream representations.
    """

    def __init__(self, params: Optional[L3_StochasticParameters] = None):
        self.params = params or L3_StochasticParameters()

        # Gene expression levels (mRNA proxy, 0-1 normalized)
        self.expression_levels = np.random.random(self.params.n_genes) * 0.3

        # Protein concentrations
        self.protein_levels = np.random.random(self.params.n_genes) * 0.2

        # Chromatin state: 0 = closed (silenced), 1 = open (active)
        self.chromatin_state = np.random.random(self.params.n_genes) > 0.5
        self.chromatin_openness = self.chromatin_state.astype(np.float32)

        # Methylation pattern (0-1, higher = more methylated = silenced)
        self.methylation = np.random.random(self.params.n_genes) * 0.3

        # Bioelectric membrane potential grid
        self.membrane_potential = np.ones(self.params.n_genes) * self.params.membrane_potential_rest

        # CISS spin state (for quantum-genomic coupling)
        self.spin_polarization = np.zeros(self.params.n_genes)

        # Regulatory network (simplified adjacency)
        self.regulatory_matrix = self._init_regulatory_network()

    def _init_regulatory_network(self) -> np.ndarray[Any, Any]:
        """Initialize gene regulatory network with sparse connections."""
        # Sparse random regulatory matrix
        matrix = np.random.random((self.params.n_genes, self.params.n_regulatory_elements))
        matrix = np.where(matrix > 0.9, matrix, 0)  # Sparse
        # Add some inhibitory connections
        matrix[:, : self.params.n_regulatory_elements // 3] *= -1
        return matrix

    def step(
        self,
        dt: float,
        l2_input: Optional[Dict[str, Any]] = None,
        bioelectric_signal: Optional[np.ndarray[Any, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Advance the layer by one time step.

        Args:
            dt: Time step in seconds.
            l2_input: Neurochemical layer output (second messengers).
            bioelectric_signal: External bioelectric modulation.

        Returns:
            Dict with expression, protein_levels, chromatin_state, output_bitstreams
        """
        # 1. Update chromatin state (epigenetic dynamics)
        # Methylation silences genes
        demeth_prob = self.params.demethylation_rate * dt
        meth_prob = self.params.methylation_rate * dt

        demeth_mask = np.random.random(self.params.n_genes) < demeth_prob
        meth_mask = np.random.random(self.params.n_genes) < meth_prob

        self.methylation = np.where(demeth_mask, self.methylation * 0.9, self.methylation)
        self.methylation = np.where(meth_mask, self.methylation + 0.1, self.methylation)
        self.methylation = np.clip(self.methylation, 0.0, 1.0)

        # Chromatin openness inversely related to methylation
        self.chromatin_openness = (
            1.0 - self.methylation + np.random.normal(0, 0.05, self.params.n_genes)
        )
        self.chromatin_openness = np.clip(self.chromatin_openness, 0.0, 1.0)

        # 2. Gene expression (stochastic transcription)
        # Only open chromatin can be transcribed
        transcription_prob = self.params.transcription_rate * self.chromatin_openness * dt
        transcription = np.random.random(self.params.n_genes) < transcription_prob

        self.expression_levels = np.where(
            transcription,
            self.expression_levels + 0.1,
            self.expression_levels - self.params.degradation_rate * dt,
        )
        self.expression_levels = np.clip(self.expression_levels, 0.0, 1.0)

        # 3. Translation to protein
        translation_prob = self.params.translation_rate * self.expression_levels * dt
        translation = np.random.random(self.params.n_genes) < translation_prob

        self.protein_levels = np.where(
            translation,
            self.protein_levels + 0.05,
            self.protein_levels - self.params.degradation_rate * dt * 0.5,
        )
        self.protein_levels = np.clip(self.protein_levels, 0.0, 1.0)

        # 4. CISS effect (quantum spin filtering)
        # Spin polarization depends on DNA chirality and electron flow
        electron_flow = np.mean(self.expression_levels)  # Proxy for metabolic activity
        self.spin_polarization = (
            self.params.ciss_efficiency * self.params.dna_chirality * electron_flow
        )
        self.spin_polarization = np.clip(
            self.spin_polarization + np.random.normal(0, 0.1, self.params.n_genes), -1.0, 1.0
        )

        # 5. Neurochemical coupling (L2 input modulates expression)
        if l2_input is not None and "second_messengers" in l2_input:
            # cAMP from second messengers activates transcription factors
            camp_level = np.mean(l2_input["second_messengers"])
            activation_boost = camp_level * self.params.neurochemical_coupling
            self.expression_levels += activation_boost * dt
            self.expression_levels = np.clip(self.expression_levels, 0.0, 1.0)

        # 6. Bioelectric pattern formation
        if bioelectric_signal is not None:
            self.membrane_potential = (
                0.9 * self.membrane_potential + 0.1 * bioelectric_signal[: self.params.n_genes]
            )
        # Internal bioelectric dynamics (gap junction diffusion)
        diffusion = np.roll(self.membrane_potential, 1) - self.membrane_potential
        self.membrane_potential += diffusion * self.params.bioelectric_coupling * dt

        # 7. Generate output bitstreams
        output_probs = self.protein_levels
        rands = np.random.random((self.params.n_genes, self.params.bitstream_length))
        output_bitstreams = (rands < output_probs[:, None]).astype(np.uint8)

        return {
            "expression_levels": self.expression_levels.copy(),
            "protein_levels": self.protein_levels.copy(),
            "chromatin_openness": self.chromatin_openness.copy(),
            "methylation": self.methylation.copy(),
            "spin_polarization": self.spin_polarization.copy(),
            "membrane_potential": self.membrane_potential.copy(),
            "output_bitstreams": output_bitstreams,
        }

    def get_global_metric(self) -> float:
        """Return the global genomic activity metric."""
        return float(np.mean(self.expression_levels))

    def get_ciss_coherence(self) -> float:
        """Return CISS spin coherence metric."""
        return float(np.abs(np.mean(self.spin_polarization)))
