# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact us: www.anulum.li  protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AFFERO GENERAL PUBLIC LICENSE v3
# Commercial Licensing: Available

"""SCPN Fusion Engine — Stochastic Colored Petri Net via DenseLayer.

Maps a Petri net (Places -> Transitions -> Places) to matrix algebra
using two ``DenseLayer`` instances for the incidence matrices.
"""

from __future__ import annotations

import numpy as np

from sc_neurocore_engine.sc_neurocore_engine import DenseLayer as _RustDenseLayer


class PetriNetEngine:
    """Stochastic Petri net engine backed by Rust ``DenseLayer``.

    Parameters
    ----------
    artifacts : dict
        Must contain:
        - ``w_in``  : (n_transitions, n_places) incidence matrix (input arcs)
        - ``w_out`` : (n_places, n_transitions) incidence matrix (output arcs)
        - ``thresholds`` : (n_transitions,) firing thresholds (0-1 probability)
        Optionally:
        - ``marking`` : (n_places,) initial token counts
    length : int
        Bitstream length for stochastic compute (default 1024).
    seed : int
        RNG seed.
    """

    def __init__(self, artifacts: dict, length: int = 1024, seed: int = 24301):
        w_in = np.asarray(artifacts["w_in"], dtype=np.float64)
        w_out = np.asarray(artifacts["w_out"], dtype=np.float64)
        thresholds = np.asarray(artifacts["thresholds"], dtype=np.float64)

        self.n_transitions, self.n_places = w_in.shape
        assert w_out.shape == (self.n_places, self.n_transitions), (
            f"w_out shape {w_out.shape} must be ({self.n_places}, {self.n_transitions})"
        )
        assert thresholds.shape == (self.n_transitions,), (
            f"thresholds shape {thresholds.shape} must be ({self.n_transitions},)"
        )

        self.thresholds = thresholds.copy()
        self.length = length

        # Layer 1: Places -> Transitions  (n_places inputs, n_transitions outputs)
        self._layer_in = _RustDenseLayer(self.n_places, self.n_transitions, length, seed)
        self._layer_in.set_weights(w_in.clip(0, 1).tolist())
        self._layer_in.refresh_packed_weights()

        # Layer 2: Transitions -> Places  (n_transitions inputs, n_places outputs)
        self._layer_out = _RustDenseLayer(self.n_transitions, self.n_places, length, seed + 1)
        self._layer_out.set_weights(w_out.clip(0, 1).tolist())
        self._layer_out.refresh_packed_weights()

        # Current marking (token counts, normalised to [0,1] for stochastic domain)
        if "marking" in artifacts:
            self._marking = np.asarray(artifacts["marking"], dtype=np.float64).copy()
        else:
            self._marking = np.zeros(self.n_places, dtype=np.float64)

        self._step_count = 0

    @property
    def marking(self) -> np.ndarray:
        """Current token marking vector."""
        return self._marking.copy()

    @property
    def step_count(self) -> int:
        return self._step_count

    def _normalise_marking(self) -> np.ndarray:
        """Map marking to [0,1] for stochastic encoding."""
        m = self._marking
        mx = m.max()
        if mx > 0:
            return m / mx
        return m

    def step(self, seed: int | None = None) -> np.ndarray:
        """Execute one Petri-net firing cycle.

        Returns
        -------
        np.ndarray
            Updated marking vector after one step.
        """
        s = seed if seed is not None else 44257 + self._step_count
        norm_marking = self._normalise_marking()

        # Forward through input arcs: marking -> transition activation
        activation = np.array(
            self._layer_in.forward(norm_marking.tolist(), s),
            dtype=np.float64,
        )

        # Apply thresholds: transitions fire if activation > threshold
        fired = (activation > self.thresholds).astype(np.float64)

        # Forward through output arcs: fired transitions -> token production
        if fired.sum() > 0:
            production = np.array(
                self._layer_out.forward(fired, s + 1),
                dtype=np.float64,
            )
        else:
            production = np.zeros(self.n_places, dtype=np.float64)

        # Update marking: consume from firing places, produce to output places
        # Simplified model: new_marking = old - consumption + production
        consumption = np.zeros(self.n_places, dtype=np.float64)
        for t in range(self.n_transitions):
            if fired[t] > 0:
                w_row = np.array(self._layer_in.get_weights()[t], dtype=np.float64)
                consumption += w_row * norm_marking

        self._marking = np.maximum(0.0, self._marking - consumption + production)
        self._step_count += 1
        return self._marking.copy()

    def run(self, n_steps: int, seed: int | None = None) -> list[np.ndarray]:
        """Run multiple steps, returning marking history."""
        history = []
        for i in range(n_steps):
            s = (seed + i) if seed is not None else None
            self.step(s)
            history.append(self._marking.copy())
        return history

    def reset(self, marking: np.ndarray | None = None):
        """Reset to initial or given marking."""
        if marking is not None:
            self._marking = np.asarray(marking, dtype=np.float64).copy()
        else:
            self._marking = np.zeros(self.n_places, dtype=np.float64)
        self._step_count = 0
