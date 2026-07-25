# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Izhikevich integrator path contracts

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.sc_izhikevich import SCIzhikevichNeuron
from tests.neuron_integrator_paths_support import as_untyped, count_spikes


def test_izhikevich_integrator_validation() -> None:
    with pytest.raises(ValueError, match="Unsupported integrator"):
        SCIzhikevichNeuron(integrator=as_untyped("bad-path"))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"a": np.nan}, "a"),
        ({"b": np.inf}, "b"),
        ({"c": np.nan}, "c"),
        ({"d": np.inf}, "d"),
        ({"dt": 0.0}, "dt"),
        ({"dt": np.nan}, "dt"),
        ({"noise_std": -0.1}, "noise_std"),
        ({"noise_std": np.nan}, "noise_std"),
    ],
)
def test_izhikevich_rejects_invalid_numerical_configuration(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        SCIzhikevichNeuron(**as_untyped(kwargs))


@pytest.mark.parametrize("integrator", ["baseline_half_euler", "rk4"])
def test_izhikevich_rejects_non_finite_input_current(integrator: str) -> None:
    neuron = SCIzhikevichNeuron(noise_std=0.0, dt=0.5, integrator=as_untyped(integrator))

    with pytest.raises(ValueError, match="input_current"):
        neuron.step(np.nan)


def test_izhikevich_rk4_regular_spiking_and_default_preserved() -> None:
    baseline = SCIzhikevichNeuron(noise_std=0.0, dt=0.5, integrator="baseline_half_euler")
    candidate = SCIzhikevichNeuron(noise_std=0.0, dt=0.5, integrator="rk4")
    default = SCIzhikevichNeuron(noise_std=0.0, dt=0.5)

    baseline_spikes = count_spikes(baseline, 10.0, 1000)
    candidate_spikes = count_spikes(candidate, 10.0, 1000)
    default_spikes = count_spikes(default, 10.0, 1000)

    assert baseline_spikes == default_spikes
    assert baseline_spikes >= 5
    assert candidate_spikes >= 5
    assert abs(candidate_spikes - baseline_spikes) <= 5


def test_izhikevich_noise_injection_is_reachable_on_both_paths() -> None:
    """``noise_std > 0`` must perturb membrane on both integrator paths."""

    for integrator in ("baseline_half_euler", "rk4"):
        neuron = SCIzhikevichNeuron(noise_std=0.5, dt=0.5, seed=42, integrator=integrator)
        v0 = neuron.v
        neuron.step(5.0)
        assert neuron.v != v0


def test_izhikevich_get_state_reflects_running_v_and_u() -> None:
    neuron = SCIzhikevichNeuron(noise_std=0.0, dt=0.5)
    state = neuron.get_state()
    assert set(state) == {"v", "u"}
    assert state["v"] == float(neuron.v)
    assert state["u"] == float(neuron.u)
