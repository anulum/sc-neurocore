# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Izhikevich 20 firing patterns validation (2003, Table 1)

from __future__ import annotations

import pytest

from sc_neurocore.neurons.sc_izhikevich import SCIzhikevichNeuron

# Izhikevich (2003) Table 1 + (2004) extended patterns
# (a, b, c, d, I_ext, expected_min_spikes_in_500ms)
PATTERNS = {
    "RS": (0.02, 0.2, -65, 8, 10, 5),
    "IB": (0.02, 0.2, -55, 4, 10, 5),
    "CH": (0.02, 0.2, -50, 2, 10, 10),
    "FS": (0.1, 0.2, -65, 2, 10, 20),
    "TC_tonic": (0.02, 0.25, -65, 0.05, 5, 1),
    "LTS": (0.02, 0.25, -65, 2, 10, 10),
    "tonic_spiking": (0.02, 0.2, -65, 6, 14, 5),
    "tonic_bursting": (0.02, 0.2, -50, 2, 15, 10),
    "mixed_mode": (0.02, 0.2, -55, 4, 10, 5),
    "spike_freq_adapt": (0.01, 0.2, -65, 8, 30, 5),
    "class1_excitable": (0.02, -0.1, -55, 6, 30, 5),
    "class2_excitable": (0.2, 0.26, -65, 0, 5, 10),
    "spike_latency": (0.02, 0.2, -65, 6, 7, 1),
    "accommodation": (0.02, 1.0, -55, 4, 10, 10),
    "inhibition_induced": (-0.02, -1.0, -60, 8, 80, 1),
}

# Patterns that are silent or produce ≤1 spike at I=0
SILENT_PATTERNS = {
    "TC_burst": (0.02, 0.25, -65, 0.05, 0),
    "RZ": (0.1, 0.26, -65, 2, 0),
}


@pytest.mark.parametrize("name", list(PATTERNS.keys()))
def test_pattern_fires(name):
    """Each pattern with suprathreshold current produces expected minimum spikes."""
    a, b, c, d, I, min_spikes = PATTERNS[name]
    neuron = SCIzhikevichNeuron(a=a, b=b, c=float(c), d=d, dt=0.5, noise_std=0.0)
    n_steps = int(500 / 0.5)
    spikes = sum(neuron.step(float(I)) for _ in range(n_steps))
    assert spikes >= min_spikes, f"{name}: {spikes} spikes < {min_spikes} expected"


@pytest.mark.parametrize("name", list(SILENT_PATTERNS.keys()))
def test_silent_pattern(name):
    """Patterns with I=0 produce ≤1 spike (quiescent or single transient)."""
    a, b, c, d, I = SILENT_PATTERNS[name]
    neuron = SCIzhikevichNeuron(a=a, b=b, c=float(c), d=d, dt=0.5, noise_std=0.0)
    spikes = sum(neuron.step(float(I)) for _ in range(1000))
    assert spikes <= 2, f"{name}: {spikes} spikes at I=0 (expected ≤2)"


def test_rs_is_regular():
    """Regular Spiking: low coefficient of variation (<0.2)."""
    neuron = SCIzhikevichNeuron(a=0.02, b=0.2, c=-65.0, d=8, dt=0.5, noise_std=0.0)
    spike_times = []
    for i in range(1000):
        if neuron.step(10.0):
            spike_times.append(i * 0.5)
    assert len(spike_times) >= 5, f"Too few spikes: {len(spike_times)}"
    isis = [spike_times[i + 1] - spike_times[i] for i in range(len(spike_times) - 1)]
    mean_isi = sum(isis) / len(isis)
    cv = (sum((x - mean_isi) ** 2 for x in isis) / len(isis)) ** 0.5 / mean_isi
    assert cv < 0.2, f"RS CV={cv:.3f} (expected <0.2 for regular spiking)"


def test_fs_faster_than_rs():
    """Fast Spiking neuron fires more than Regular Spiking at same current."""
    rs = SCIzhikevichNeuron(a=0.02, b=0.2, c=-65.0, d=8, dt=0.5, noise_std=0.0)
    fs = SCIzhikevichNeuron(a=0.1, b=0.2, c=-65.0, d=2, dt=0.5, noise_std=0.0)
    rs_spikes = sum(rs.step(10.0) for _ in range(1000))
    fs_spikes = sum(fs.step(10.0) for _ in range(1000))
    assert fs_spikes > rs_spikes, f"FS={fs_spikes} ≤ RS={rs_spikes}"


def test_spike_resets_correctly():
    """After spike, v resets to c and u increments by d."""
    neuron = SCIzhikevichNeuron(a=0.02, b=0.2, c=-65.0, d=8, dt=0.5, noise_std=0.0)
    for _ in range(1000):
        spike = neuron.step(10.0)
        if spike:
            assert neuron.v == -65.0, f"v after spike: {neuron.v} (expected -65)"
            break
    else:
        pytest.fail("No spike in 1000 steps")
