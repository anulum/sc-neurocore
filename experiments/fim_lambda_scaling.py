# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Experiment: FIM λ_c scaling law verification

"""Verify quantum-control prediction: λ_c = 0.149 · N^1.02.

For each network size N, sweep fim_lambda and measure the mean
firing rate standard deviation (proxy for coherence — lower std
= more synchronised). Find λ_c as the λ where std drops below
a threshold.

This is a NECESSARY CONDITION test. The Kuramoto λ_c was derived
for phase oscillators; the LIF network may have a different
scaling law. If it matches, the Kuramoto→LIF mapping is valid.
If it diverges, LIF dynamics add corrections to the mean-field theory.

Usage:
    python experiments/fim_lambda_scaling.py
"""

import numpy as np

from sc_neurocore import StochasticLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


def measure_coherence(n_neurons: int, fim_lambda: float, duration: float = 0.5) -> float:
    """Run network and return firing rate CV (lower = more coherent)."""
    pop = Population(StochasticLIFNeuron, n=n_neurons, label="exc")
    proj = Projection(pop, pop, weight=0.05, probability=0.1, seed=42)
    drive = PoissonInput(n=n_neurons, rate_hz=50.0, weight=2.0, dt=0.001, seed=42)
    mon = SpikeMonitor(pop, label="spk")
    net = Network(pop, proj, drive, mon, fim_lambda=fim_lambda)
    net.run(duration=duration, dt=0.001)

    # Per-neuron firing rate
    trains = mon.spike_trains
    rates = np.zeros(n_neurons)
    for nid, times in trains.items():
        rates[nid] = len(times) / duration

    mean_rate = np.mean(rates)
    if mean_rate < 1.0:
        return float("inf")  # too sparse to measure coherence
    return float(np.std(rates) / mean_rate)  # CV of rates


def find_lambda_c(n_neurons: int, threshold_cv: float = 0.5) -> float:
    """Binary search for λ_c where rate CV drops below threshold."""
    lo, hi = 0.0, 5.0 * n_neurons
    for _ in range(15):
        mid = (lo + hi) / 2
        cv = measure_coherence(n_neurons, mid)
        if cv < threshold_cv:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def main() -> None:
    print("FIM λ_c Scaling Law Verification")
    print("Quantum-control prediction: λ_c = 0.149 · N^1.02")
    print()
    print(f"{'N':>6s}  {'λ_c (measured)':>14s}  {'λ_c (predicted)':>15s}  {'ratio':>8s}")
    print("-" * 48)

    results = []
    for n in [30, 50, 100, 200]:
        predicted = 0.149 * n**1.02
        measured = find_lambda_c(n)
        ratio = measured / predicted if predicted > 0 else float("inf")
        results.append((n, measured, predicted, ratio))
        print(f"{n:6d}  {measured:14.2f}  {predicted:15.2f}  {ratio:8.2f}")

    print()
    print("If ratio ≈ constant: LIF scaling matches Kuramoto (mapping valid)")
    print("If ratio varies with N: LIF has different scaling (corrections needed)")


if __name__ == "__main__":
    main()
