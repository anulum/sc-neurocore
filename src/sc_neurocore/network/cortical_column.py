# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Canonical microcircuit: 6-layer cortical column

"""Canonical cortical microcircuit (Douglas & Martin 2004, Potjans & Diesmann 2014).

Implements the canonical 4-population cortical column:
- L2/3: Superficial pyramidal (excitatory) + interneurons (inhibitory)
- L4: Spiny stellate (thalamic input relay)
- L5: Deep pyramidal (cortical output, motor commands)
- L6: Corticothalamic (feedback to thalamus)

Connectivity follows the canonical pattern:
- Thalamus → L4 (feedforward input)
- L4 → L2/3 (feedforward drive)
- L2/3 → L5 (top-down drive)
- L5 → L6 (deep output)
- L6 → L4 (feedback modulation)
- L2/3 ↔ L2/3 (lateral recurrence + inhibition)

    from sc_neurocore.network.cortical_column import CorticalColumn

    col = CorticalColumn(n_per_layer=20)
    output = col.step(thalamic_input)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CorticalColumn:
    """Canonical 4-population cortical microcircuit.

    Parameters
    ----------
    n_per_layer : int
        Number of neurons per layer.
    tau : float
        Membrane time constant (ms).
    dt : float
        Timestep (ms).
    w_exc : float
        Excitatory connection strength.
    w_inh : float
        Inhibitory connection strength.
    threshold : float
        Firing threshold.
    seed : int or None
        Random seed for weight initialization.
    """

    n_per_layer: int = 20
    tau: float = 10.0
    dt: float = 1.0
    w_exc: float = 0.1
    w_inh: float = -0.15
    threshold: float = 1.0
    seed: int | None = None

    def __post_init__(self):
        n = self.n_per_layer
        rng = np.random.RandomState(self.seed)
        decay = np.exp(-self.dt / self.tau)
        self._decay = decay

        # Membrane voltages per layer
        self.v_l23_exc = np.zeros(n)
        self.v_l23_inh = np.zeros(n)
        self.v_l4 = np.zeros(n)
        self.v_l5 = np.zeros(n)
        self.v_l6 = np.zeros(n)

        # Connection weight matrices (sparse, random)
        def _make_weights(n_pre, n_post, strength, prob=0.3):
            w = rng.uniform(0, abs(strength), (n_post, n_pre))
            mask = rng.random((n_post, n_pre)) < prob
            w *= mask
            if strength < 0:
                w = -w
            return w

        # Canonical connectivity
        self.w_thal_to_l4 = _make_weights(n, n, self.w_exc, 0.5)
        self.w_l4_to_l23e = _make_weights(n, n, self.w_exc, 0.4)
        self.w_l23e_to_l23i = _make_weights(n, n, self.w_exc, 0.3)
        self.w_l23i_to_l23e = _make_weights(n, n, self.w_inh, 0.3)
        self.w_l23e_to_l5 = _make_weights(n, n, self.w_exc, 0.3)
        self.w_l5_to_l6 = _make_weights(n, n, self.w_exc, 0.3)
        self.w_l6_to_l4 = _make_weights(n, n, self.w_exc * 0.5, 0.2)

    def step(self, thalamic_input: np.ndarray) -> dict[str, np.ndarray]:
        """Advance one timestep.

        Parameters
        ----------
        thalamic_input : np.ndarray
            Shape (n_per_layer,) — feedforward drive from thalamus.

        Returns
        -------
        dict with keys 'l23_exc', 'l23_inh', 'l4', 'l5', 'l6',
        each containing binary spike vectors of shape (n_per_layer,).
        """
        thal = np.atleast_1d(np.asarray(thalamic_input, dtype=np.float64))

        # L4: thalamic input + L6 feedback
        i_l4 = self.w_thal_to_l4 @ thal + self.w_l6_to_l4 @ (self.v_l6 > self.threshold).astype(
            float
        )
        self.v_l4 = self._decay * self.v_l4 + i_l4 * self.dt / self.tau
        spk_l4 = (self.v_l4 > self.threshold).astype(np.float64)
        self.v_l4 -= spk_l4 * self.threshold

        # L2/3 excitatory: L4 feedforward + L2/3 inhibitory feedback
        i_l23e = self.w_l4_to_l23e @ spk_l4 + self.w_l23i_to_l23e @ (
            self.v_l23_inh > self.threshold
        ).astype(float)
        self.v_l23_exc = self._decay * self.v_l23_exc + i_l23e * self.dt / self.tau
        spk_l23e = (self.v_l23_exc > self.threshold).astype(np.float64)
        self.v_l23_exc -= spk_l23e * self.threshold

        # L2/3 inhibitory: driven by L2/3 excitatory
        i_l23i = self.w_l23e_to_l23i @ spk_l23e
        self.v_l23_inh = self._decay * self.v_l23_inh + i_l23i * self.dt / self.tau
        spk_l23i = (self.v_l23_inh > self.threshold).astype(np.float64)
        self.v_l23_inh -= spk_l23i * self.threshold

        # L5: driven by L2/3 excitatory (cortical output)
        i_l5 = self.w_l23e_to_l5 @ spk_l23e
        self.v_l5 = self._decay * self.v_l5 + i_l5 * self.dt / self.tau
        spk_l5 = (self.v_l5 > self.threshold).astype(np.float64)
        self.v_l5 -= spk_l5 * self.threshold

        # L6: driven by L5 (corticothalamic feedback)
        i_l6 = self.w_l5_to_l6 @ spk_l5
        self.v_l6 = self._decay * self.v_l6 + i_l6 * self.dt / self.tau
        spk_l6 = (self.v_l6 > self.threshold).astype(np.float64)
        self.v_l6 -= spk_l6 * self.threshold

        return {
            "l23_exc": spk_l23e,
            "l23_inh": spk_l23i,
            "l4": spk_l4,
            "l5": spk_l5,
            "l6": spk_l6,
        }

    def run(self, thalamic_input: np.ndarray, steps: int = 100) -> dict[str, np.ndarray]:
        """Run for multiple timesteps with constant input.

        Returns dict mapping layer name → (steps, n_per_layer) spike arrays.
        """
        results = {k: [] for k in ("l23_exc", "l23_inh", "l4", "l5", "l6")}
        for _ in range(steps):
            spikes = self.step(thalamic_input)
            for k, v in spikes.items():
                results[k].append(v.copy())
        return {k: np.array(v) for k, v in results.items()}

    def reset(self):
        self.v_l23_exc[:] = 0
        self.v_l23_inh[:] = 0
        self.v_l4[:] = 0
        self.v_l5[:] = 0
        self.v_l6[:] = 0
