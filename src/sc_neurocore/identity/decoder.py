# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike state to context extraction for session priming

"""Extract cognitive state from spiking network for session priming."""

from __future__ import annotations

from typing import Any

import numpy as np

from sc_neurocore._source_checkout import require_source_checkout_package
from sc_neurocore.spike_train_statistics import cv_isi, firing_rate

from .substrate import IdentitySubstrate

# The decoder's PCA and connectivity measures come from sc_neurocore.analysis,
# which only a source checkout ships. It is imported where it is used, so that
# sc_neurocore.identity stays importable from an installed wheel.
_ANALYSIS = "sc_neurocore.analysis"


class StateDecoder:
    """Extract cognitive state from spiking network for session priming."""

    def __init__(self, substrate: IdentitySubstrate) -> None:
        """Read state from ``substrate``."""
        self.substrate = substrate

    def _recent_trains(self, n_neurons: int = 50, window: int = 1000) -> list[np.ndarray[Any, Any]]:
        """Get recent binary spike trains for the first n_neurons."""
        history = self.substrate.spike_history
        if len(history) < 2:
            return []
        recent = history[-window:]
        n = min(n_neurons, self.substrate.n_cortical)
        return [np.array([h[i] for h in recent], dtype=np.int8) for i in range(n)]

    def extract_dominant_patterns(self, n_components: int = 10) -> np.ndarray[Any, Any]:
        """PCA on recent spike trains -> dominant activity patterns."""
        trains = self._recent_trains()
        if not trains:
            return np.zeros((0, 0))
        require_source_checkout_package(_ANALYSIS, "StateDecoder.extract_dominant_patterns")
        from sc_neurocore.analysis import spike_train_pca

        n_comp = min(n_components, len(trains))
        projected, _ = spike_train_pca(trains, n_components=n_comp)
        return projected

    def extract_attractor_states(self, threshold: float = 0.8) -> list[np.ndarray[Any, Any]]:
        """Find stable attractor states via correlation clustering.

        Groups of neurons that fire together with correlation above
        threshold are identified as attractor ensembles.
        """
        trains = self._recent_trains(n_neurons=30)
        if len(trains) < 3:
            return []
        require_source_checkout_package(_ANALYSIS, "StateDecoder.extract_attractor_states")
        from sc_neurocore.analysis import functional_connectivity

        fc = functional_connectivity(trains)
        n = fc.shape[0]
        visited: set[int] = set()
        attractors: list[np.ndarray[Any, Any]] = []

        for i in range(n):
            if i in visited:
                continue
            group = [i]
            for j in range(i + 1, n):
                if fc[i, j] >= threshold:
                    group.append(j)
                    visited.add(j)
            if len(group) >= 2:
                visited.add(i)
                attractors.append(np.array(group, dtype=np.int64))

        return attractors

    def extract_connectivity_signature(self) -> np.ndarray[Any, Any]:
        """Functional connectivity matrix summarizing learned structure."""
        trains = self._recent_trains(n_neurons=30)
        if not trains:
            return np.zeros((0, 0))
        require_source_checkout_package(_ANALYSIS, "StateDecoder.extract_connectivity_signature")
        from sc_neurocore.analysis import functional_connectivity

        return functional_connectivity(trains)

    def generate_priming_context(self) -> str:
        """Generate a text summary of current network state."""
        history = self.substrate.spike_history
        n_steps = len(history)

        if n_steps < 10:
            return f"Substrate dormant. {n_steps} steps recorded. No patterns yet."
        require_source_checkout_package(_ANALYSIS, "StateDecoder.generate_priming_context")

        patterns = self.extract_dominant_patterns(n_components=5)
        n_patterns = patterns.shape[0] if patterns.ndim == 2 else 0

        attractors = self.extract_attractor_states()
        n_attractors = len(attractors)

        trains = self._recent_trains(n_neurons=20)
        rates = [firing_rate(t) for t in trains] if trains else []
        mean_rate = float(np.mean(rates)) if rates else 0.0

        cvs = [cv_isi(t) for t in trains] if trains else []
        valid_cvs = [c for c in cvs if not np.isnan(c)]
        mean_cv = float(np.mean(valid_cvs)) if valid_cvs else float("nan")

        health = self.substrate.health_check()

        lines = [
            f"Substrate active: {n_steps} steps.",
            f"Dominant patterns: {n_patterns}.",
            f"Stable attractors: {n_attractors}"
            + (f" (sizes: {[len(a) for a in attractors]})." if attractors else "."),
            f"Mean rate: {mean_rate:.1f} Hz, CV: {mean_cv:.2f}.",
            f"Health: {'OK' if health['is_healthy'] else 'DEGRADED'}.",
        ]

        ee_weights = self.substrate.ee_weights
        if ee_weights.size > 0:
            w_mean = float(ee_weights.mean())
            w_std = float(ee_weights.std())
            lines.append(f"E-E weights: mean={w_mean:.4f}, std={w_std:.4f}.")

        return " ".join(lines)
