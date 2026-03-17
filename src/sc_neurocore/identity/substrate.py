# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Persistent spiking network for identity continuity

"""Persistent spiking neural network for identity continuity.

Maintains continuous membrane state across sessions. STDP and
homeostatic plasticity modify weights based on experience.
The network accumulates identity as stable attractors in the
cortical-inhibitory-memory architecture.
"""

from __future__ import annotations

import numpy as np

from sc_neurocore.network import (
    Population,
    Projection,
    Network,
    SpikeMonitor,
)
from sc_neurocore.network.topology import small_world
from sc_neurocore.analysis import (
    firing_rate,
    cv_isi,
    fano_factor,
    power_spectrum,
    spike_train_pca,
    functional_connectivity,
)

_HOMEOSTATIC_TARGET_RATE = 10.0  # Hz, Turrigiano 2008


class IdentitySubstrate:
    """Persistent spiking neural network for identity continuity.

    Three populations:
    - cortical: HodgkinHuxley excitatory (main processing)
    - inhibitory: WangBuzsaki fast-spiking (balance/stability)
    - memory: HindmarshRose bursting (pattern storage via attractors)

    Connectivity: small-world E->E with STDP, random E->I, I->E, E->M, M->E.
    """

    def __init__(self, n_cortical=500, n_inhibitory=200, n_memory=100, seed=42):
        self.n_cortical = n_cortical
        self.n_inhibitory = n_inhibitory
        self.n_memory = n_memory
        self.seed = seed

        self.cortical = Population("HodgkinHuxleyNeuron", n_cortical, label="HodgkinHuxleyNeuron")
        self.inhibitory = Population("WangBuzsakiNeuron", n_inhibitory, label="WangBuzsakiNeuron")
        self.memory = Population("HindmarshRoseNeuron", n_memory, label="HindmarshRoseNeuron")

        self._build_projections(seed)
        self._build_monitors()
        self._build_network()

        self._spike_history: list[np.ndarray] = []
        self._total_steps = 0

    def _build_projections(self, seed):
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 2**31, size=6)

        # E->E: small-world with STDP
        n_c = self.n_cortical
        sw_csr = small_world(n_c, k=6, p_rewire=0.1, weight=0.5, seed=int(seeds[0]))
        self.proj_ee = Projection(
            self.cortical,
            self.cortical,
            weight=0.5,
            topology=sw_csr,
            plasticity="stdp",
            seed=int(seeds[0]),
        )

        # E->I: random excitatory drive to inhibitory
        self.proj_ei = Projection(
            self.cortical,
            self.inhibitory,
            weight=0.8,
            probability=0.2,
            topology="random",
            seed=int(seeds[1]),
        )

        # I->E: inhibitory feedback (negative weight)
        self.proj_ie = Projection(
            self.inhibitory,
            self.cortical,
            weight=-1.0,
            probability=0.3,
            topology="random",
            seed=int(seeds[2]),
        )

        # E->M: cortical drives memory (pattern imprinting)
        self.proj_em = Projection(
            self.cortical,
            self.memory,
            weight=0.6,
            probability=0.15,
            topology="random",
            seed=int(seeds[3]),
        )

        # M->E: memory reactivation drives cortex
        self.proj_me = Projection(
            self.memory,
            self.cortical,
            weight=0.4,
            probability=0.1,
            topology="random",
            seed=int(seeds[4]),
        )

        # I->I: mutual inhibition for competition
        self.proj_ii = Projection(
            self.inhibitory,
            self.inhibitory,
            weight=-0.5,
            probability=0.15,
            topology="random",
            seed=int(seeds[5]),
        )

    def _build_monitors(self):
        self.mon_cortical = SpikeMonitor(self.cortical)
        self.mon_inhibitory = SpikeMonitor(self.inhibitory)
        self.mon_memory = SpikeMonitor(self.memory)

    def _build_network(self):
        self.network = Network(
            self.cortical,
            self.inhibitory,
            self.memory,
            self.proj_ee,
            self.proj_ei,
            self.proj_ie,
            self.proj_em,
            self.proj_me,
            self.proj_ii,
            self.mon_cortical,
            self.mon_inhibitory,
            self.mon_memory,
            seed=self.seed,
        )

    def step(self, stimuli=None, dt=0.001):
        """Advance one timestep. Inject external current into cortical neurons."""
        if stimuli is not None:
            currents = np.asarray(stimuli, dtype=np.float64)
            if currents.shape[0] < self.n_cortical:
                padded = np.zeros(self.n_cortical, dtype=np.float64)
                padded[: currents.shape[0]] = currents
                currents = padded
        else:
            currents = np.zeros(self.n_cortical, dtype=np.float64)

        spikes_c = self.cortical.step_all(currents)
        i_from_c = self.proj_ei.propagate(spikes_c)
        i_from_i_to_e = self.proj_ie.propagate(np.zeros(self.n_inhibitory, dtype=np.int8))
        spikes_i = self.inhibitory.step_all(i_from_c)

        i_feedback = self.proj_ie.propagate(spikes_i)
        i_from_m = self.proj_me.propagate(np.zeros(self.n_memory, dtype=np.int8))
        i_to_m = self.proj_em.propagate(spikes_c)
        spikes_m = self.memory.step_all(i_to_m)

        self.proj_ee.update_plasticity(spikes_c, spikes_c)
        self._spike_history.append(spikes_c.copy())
        self._total_steps += 1
        return spikes_c

    def run(self, duration, dt=0.001, stimuli_sequence=None):
        """Run for *duration* seconds. Optional time-varying stimuli array.

        stimuli_sequence: (n_steps, n_cortical) array or None.
        """
        n_steps = int(round(duration / dt))
        all_spikes = np.zeros((n_steps, self.n_cortical), dtype=np.int8)
        for t in range(n_steps):
            stim = stimuli_sequence[t] if stimuli_sequence is not None else None
            all_spikes[t] = self.step(stim, dt)
        return all_spikes

    def inject_experience(self, reasoning_trace: str):
        """Encode a reasoning trace as spike patterns and inject via run().

        Uses TraceEncoder (imported lazily to avoid circular deps).
        """
        from .encoder import TraceEncoder

        encoder = TraceEncoder(n_neurons=self.n_cortical, seed=self.seed)
        pattern = encoder.encode(reasoning_trace, duration_ms=200, dt=0.001)
        n_steps = pattern.shape[1]
        for t in range(n_steps):
            currents = pattern[:, t] * 15.0  # scale spikes to nA-range current
            self.step(currents)

    def extract_state(self) -> dict:
        """Extract current network state for session priming."""
        if len(self._spike_history) < 10:
            return {
                "firing_rates": np.zeros(self.n_cortical),
                "dominant_patterns": np.zeros((0, 0)),
                "explained_variance": np.array([]),
                "connectivity": np.zeros((0, 0)),
                "total_steps": self._total_steps,
            }

        trains = [
            np.array([h[i] for h in self._spike_history[-1000:]], dtype=np.int8)
            for i in range(min(self.n_cortical, 50))
        ]

        rates = np.array([firing_rate(t) for t in trains])
        projected, explained = spike_train_pca(trains, n_components=min(5, len(trains)))

        n_fc = min(20, len(trains))
        fc = functional_connectivity(trains[:n_fc])

        return {
            "firing_rates": rates,
            "dominant_patterns": projected,
            "explained_variance": explained,
            "connectivity": fc,
            "total_steps": self._total_steps,
        }

    def health_check(self) -> dict:
        """L16 Director: check network dynamics are healthy."""
        if len(self._spike_history) < 100:
            return {
                "mean_rate": 0.0,
                "cv": float("nan"),
                "fano": float("nan"),
                "spectral_entropy": float("nan"),
                "is_healthy": True,
                "n_steps": self._total_steps,
            }

        recent = np.array(self._spike_history[-1000:], dtype=np.int8)
        pop_train = recent.sum(axis=1).astype(np.int8)
        pop_train_binary = (pop_train > 0).astype(np.int8)

        mean_r = firing_rate(pop_train_binary)
        cv = cv_isi(pop_train_binary)
        fano = fano_factor(pop_train_binary, window_ms=50.0)
        psd, freqs = power_spectrum(pop_train_binary)
        if psd.size > 0 and psd.sum() > 0:
            p_norm = psd / psd.sum()
            p_norm = p_norm[p_norm > 0]
            s_entropy = float(-np.sum(p_norm * np.log2(p_norm)))
        else:
            s_entropy = 0.0

        rate_ok = 1.0 <= mean_r <= 500.0
        cv_ok = np.isnan(cv) or 0.2 <= cv <= 3.0
        fano_ok = np.isnan(fano) or 0.1 <= fano <= 10.0

        return {
            "mean_rate": mean_r,
            "cv": cv,
            "fano": fano,
            "spectral_entropy": s_entropy,
            "is_healthy": rate_ok and cv_ok and fano_ok,
            "n_steps": self._total_steps,
        }

    @property
    def spike_history(self) -> list[np.ndarray]:
        return self._spike_history

    @property
    def ee_weights(self) -> np.ndarray:
        return self.proj_ee.data.copy()
