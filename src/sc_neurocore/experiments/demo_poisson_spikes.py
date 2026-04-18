# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Demo Poisson Spikes

from __future__ import annotations
import numpy as np
from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron


def run_demo() -> None:
    neuron = StochasticLIFNeuron(
        v_rest=0.0,
        v_reset=0.0,
        v_threshold=1.0,
        tau_mem=20.0,
        dt=1.0,
        noise_std=0.1,
        resistance=1.0,
        seed=42,
    )

    T = 2000
    I = 0.06 * np.ones(T)
    spikes = np.zeros(T, dtype=int)

    for t in range(T):
        spikes[t] = neuron.step(I[t])

    rate_hz = spikes.sum() / (T * neuron.dt) * 1000.0
    print(f"Total spikes: {spikes.sum()}, firing rate ≈ {rate_hz:.2f} Hz")


if __name__ == "__main__":
    run_demo()
