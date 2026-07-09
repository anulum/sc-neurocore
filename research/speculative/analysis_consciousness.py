# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


import numpy as np


class PhiEvaluator:
    """
    Evaluates Integrated Information (Phi) for SC Networks.
    Based on 'PyPhi' principles but adapted for bitstreams.
    """

    @staticmethod
    def entropy(bitstream: np.ndarray) -> float:
        """Shannon entropy of a bitstream distribution."""
        p1 = np.mean(bitstream)
        p0 = 1.0 - p1
        if p0 <= 0 or p1 <= 0:
            return 0.0
        return -(p1 * np.log2(p1) + p0 * np.log2(p0))

    @staticmethod
    def calculate_phi(layer_outputs: np.ndarray) -> float:
        """
        Approximates Phi for a layer state.
        Phi ~ Integration - Differentiation.
        Phi = H(Whole) - Sum(H(Parts)) ??

        IT 3.0 is complex. We use a simple heuristic:
        Phi = Total Correlation - Integration?

        We use 'Interaction Information':
        I(X;Y) = H(X) + H(Y) - H(X,Y)

        Phi ~ Sum(H(Neurons)) - H(System)
        If system is fully independent, H(System) = Sum(H). Phi = 0.
        If system is integrated, H(System) < Sum(H). Phi > 0.
        """
        # layer_outputs: (n_neurons,) probabilities or bitstreams
        # We need bitstreams to calculate joint entropy.
        # Assuming we have access to history or we treat prob as state.

        # Let's assume input is (n_neurons, length) bitstreams
        if layer_outputs.ndim == 1:
            return 0.0  # Snapshot has no entropy

        n_neurons, length = layer_outputs.shape

        # 1. Sum of Individual Entropies
        sum_h_parts = 0.0
        for i in range(n_neurons):
            sum_h_parts += PhiEvaluator.entropy(layer_outputs[i])

        # 2. Joint Entropy
        # We treat each time step as a state vector.
        # Count unique states.
        # Pack bits to int for easy counting
        # (Assuming n_neurons < 64)

        states = np.zeros(length, dtype=np.uint64)
        for i in range(n_neurons):
            states |= layer_outputs[i].astype(np.uint64) << np.uint64(i)

        uniques, counts = np.unique(states, return_counts=True)
        probs = counts / length
        h_whole = -np.sum(probs * np.log2(probs))

        phi = sum_h_parts - h_whole
        return max(0.0, phi)
