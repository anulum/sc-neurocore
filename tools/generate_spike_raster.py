# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Generate a publication-quality spike raster PNG from

"""Generate a publication-quality spike raster PNG from StochasticLIFNeuron."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron

N_NEURONS = 5
N_STEPS = 200
COLORS = ["#0077B6", "#E63946", "#2A9D8F", "#E9C46A", "#7209B7"]

neurons = [
    StochasticLIFNeuron(tau_mem=10.0, noise_std=0.05, resistance=1.0, seed=i)
    for i in range(N_NEURONS)
]

spikes: list[list[int]] = [[] for _ in range(N_NEURONS)]
for t in range(N_STEPS):
    for i, neuron in enumerate(neurons):
        phase = 2 * np.pi * i / N_NEURONS
        current = 0.12 + 0.06 * np.sin(2 * np.pi * t / 50 + phase)
        if neuron.step(current):
            spikes[i].append(t)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 2.5), dpi=300)
for i, times in enumerate(spikes):
    ax.eventplot(times, lineoffsets=i, linelengths=0.7, colors=COLORS[i], linewidths=0.8)

ax.set_xlim(0, N_STEPS)
ax.set_ylim(-0.5, N_NEURONS - 0.5)
ax.set_yticks(range(N_NEURONS))
ax.set_xlabel("Time Step", fontsize=9)
ax.set_ylabel("Neuron", fontsize=9)
ax.set_title("SC-NeuroCore \u2014 LIF Spike Raster", fontsize=10, fontweight="bold")
ax.tick_params(labelsize=8)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(False)

out = ROOT / "docs" / "assets" / "spike_raster.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved {out.relative_to(ROOT)}")
