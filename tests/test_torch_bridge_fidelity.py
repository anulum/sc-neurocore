# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network.to_torch(): the bridge steps as the models step, or refuses

"""``Network.to_torch()`` reproduces each supported model's own step, per neuron.

Reproducers of RIGOR review 271: the bridge stepped an Euler LIF with a strict
threshold and a subtractive reset where the models take an exact flow (SC
Lapicque) or forward Euler (StochasticLIF) with ``>=`` and a hard reset; it
substituted the SC profile for the 1907 source profile; it gave every neuron
the first neuron's voltage; it overwrote parallel CSR entries and ignored the
weight threshold; and float64 failed on the second step. Each spike train here
is compared with stepping the Python neurons themselves.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.network.network import Network
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection

torch = pytest.importorskip("torch")


def _python_trains(population: Population, drive: np.ndarray) -> np.ndarray:
    """Step each neuron of ``population`` through ``drive[t, i]`` and return its spikes."""
    trains = np.zeros(drive.shape, dtype=np.float64)
    for t in range(drive.shape[0]):
        for i, neuron in enumerate(population.neurons):
            trains[t, i] = neuron.step(float(drive[t, i]))
    return trains


def _bridge_trains(population: Population, drive: np.ndarray, dtype: object = None) -> np.ndarray:
    """Run a one-population bridge on ``drive`` and return its spike traces."""
    bridge = Network(population).to_torch()
    inputs = torch.tensor(drive[:, None, :], dtype=dtype or torch.float32)
    if dtype is not None:
        bridge = bridge.to(dtype=dtype)
    _counts, traces = bridge(inputs, return_traces=True)
    return np.asarray(traces[population.label][:, 0, :].detach().numpy(), dtype=np.float64)


@pytest.mark.parametrize(
    ("model", "drive_value"),
    [("SCLapicqueLIFNeuron", 1.5), ("SCLapicqueLIFNeuron", 1.0), ("StochasticLIFNeuron", 1.0)],
)
def test_spike_trains_equal_the_models(model: str, drive_value: float) -> None:
    """Exact flow / Euler update, >= threshold and hard reset: bridge = Python, with spikes."""
    drive = np.full((30, 3), drive_value)
    drive[:, 1] *= 2.0
    expected = _python_trains(Population(model, 3, params={"dt": 1.0}), drive)
    assert np.array_equal(_bridge_trains(Population(model, 3, params={"dt": 1.0}), drive), expected)
    assert expected.sum() > 0


def test_hard_reset_after_a_large_input() -> None:
    """StochasticLIF input [3, 0, 0]: one spike, then the reset voltage, not a subtraction."""
    drive = np.array([[3.0], [0.0], [0.0]])
    expected = _python_trains(Population("StochasticLIFNeuron", 1), drive)
    assert np.array_equal(_bridge_trains(Population("StochasticLIFNeuron", 1), drive), expected)
    assert expected[:, 0].tolist() == [1.0, 0.0, 0.0]


def test_each_neuron_starts_from_its_own_voltage() -> None:
    """set_voltages([0, 0.99]) then one 0.3 step: the second neuron spikes in both."""
    drive = np.array([[0.3, 0.3]])
    reference = Population("StochasticLIFNeuron", 2)
    reference.set_voltages(np.array([0.0, 0.99]))
    expected = _python_trains(reference, drive)
    population = Population("StochasticLIFNeuron", 2)
    population.set_voltages(np.array([0.0, 0.99]))
    assert np.array_equal(_bridge_trains(population, drive), expected)
    assert expected[0].tolist() == [0.0, 1.0]


def test_the_1907_source_profile_is_refused() -> None:
    """The source profile latches one event and has no reset; the bridge does not substitute."""
    with pytest.raises(NotImplementedError, match="lapicque_1907"):
        Network(Population("LapicqueNeuron", 1, params={"dt": 1.0})).to_torch()


def test_parallel_connections_and_weight_threshold_are_refused() -> None:
    """Two CSR entries for one pair, or a pruning threshold, have no lossless dense form."""
    source = Population("StochasticLIFNeuron", 1)
    target = Population("StochasticLIFNeuron", 2)
    parallel = (np.array([0, 2]), np.array([1, 1]), np.array([0.4, 0.8]))
    with pytest.raises(NotImplementedError, match="parallel"):
        Network(source, target, Projection(source, target, 1.0, topology=parallel)).to_torch()
    single = (np.array([0, 1]), np.array([1]), np.array([1.2]))
    pruned = Projection(source, target, 1.0, topology=single, weight_threshold=1.3)
    with pytest.raises(NotImplementedError, match="weight_threshold"):
        Network(source, target, pruned).to_torch()


def test_untrained_sync_keeps_every_weight() -> None:
    """A round trip through the bridge without training leaves the CSR weights unchanged."""
    source = Population("StochasticLIFNeuron", 2)
    target = Population("StochasticLIFNeuron", 2)
    csr = (np.array([0, 2, 3]), np.array([0, 1, 1]), np.array([0.4, 0.8, 0.6]))
    projection = Projection(source, target, 1.0, topology=csr)
    before = projection.data.copy()
    Network(source, target, projection).to_torch().sync_to_network()
    assert np.allclose(projection.data, before)


def test_float64_runs_several_projected_steps() -> None:
    """A double bridge with double input keeps its dtype through the spikes of every step."""
    source = Population("StochasticLIFNeuron", 2)
    target = Population("StochasticLIFNeuron", 2)
    projection = Projection(source, target, 0.8, topology="all_to_all")
    bridge = Network(source, target, projection).to_torch().to(dtype=torch.float64)
    counts = bridge(torch.full((5, 1, 2), 2.0, dtype=torch.float64))
    assert counts.dtype == torch.float64 and counts.sum().item() > 0
    counts.sum().backward()
