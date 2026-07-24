# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (runtime_atomic_rejects) from former test_model_resonate_and_fire.py

from __future__ import annotations

from tests.model_resonate_and_fire_support import *  # noqa: F403

@pytest.mark.parametrize("current", (np.nan, np.inf, -np.inf, object()))
def test_invalid_current_is_atomic(current: object) -> None:
    neuron = ResonateAndFireNeuron(x=0.25, y=-0.5)
    before = (neuron.x, neuron.y)
    with pytest.raises(ValueError, match="current"):
        neuron.step(cast(float, current))
    assert (neuron.x, neuron.y) == before


def test_corrupted_runtime_configuration_is_atomic() -> None:
    neuron = ResonateAndFireNeuron(x=0.25, y=-0.5)
    neuron.dt = 0.0
    before = (neuron.x, neuron.y)
    with pytest.raises(ValueError, match="dt"):
        neuron.step(0.5)
    assert (neuron.x, neuron.y) == before


@pytest.mark.parametrize("field", ("x", "y"))
def test_corrupted_runtime_state_type_is_atomic(field: str) -> None:
    neuron = ResonateAndFireNeuron(x=0.25, y=-0.5)
    setattr(neuron, field, object())
    before = (neuron.x, neuron.y)
    with pytest.raises(ValueError, match="state must be numeric"):
        neuron.step(0.5)
    assert (neuron.x, neuron.y) == before


@pytest.mark.parametrize("field", ("b", "omega", "threshold", "dt"))
def test_corrupted_runtime_parameter_type_is_atomic(field: str) -> None:
    neuron = ResonateAndFireNeuron(x=0.25, y=-0.5)
    setattr(neuron, field, object())
    before = (neuron.x, neuron.y)
    with pytest.raises(ValueError, match="parameters must be numeric"):
        neuron.step(0.5)
    assert (neuron.x, neuron.y) == before


def test_nonfinite_exact_candidate_is_atomic() -> None:
    neuron = ResonateAndFireNeuron(
        x=0.25,
        y=-0.5,
        b=1.0e308,
        omega=1.0,
        threshold=1.0e308,
        dt=1.0e308,
    )
    before = (neuron.x, neuron.y)
    with pytest.raises(FloatingPointError, match="coefficients"):
        neuron.step(1.0e308)
    assert (neuron.x, neuron.y) == before


def test_batch_rejects_nonfinite_drive_before_mutation() -> None:
    neuron = ResonateAndFireNeuron(x=0.25, y=-0.5)
    before = (neuron.x, neuron.y)
    with pytest.raises(ValueError, match="finite"):
        neuron.simulate([0.0, np.nan, 1.0], backend="python")
    assert (neuron.x, neuron.y) == before


def test_batch_rejects_unknown_backend_before_mutation() -> None:
    neuron = ResonateAndFireNeuron(x=0.25, y=-0.5)
    before = (neuron.x, neuron.y)
    with pytest.raises(ValueError, match="unknown resonate-and-fire backend"):
        neuron.simulate([0.0], backend="fortran")
    assert (neuron.x, neuron.y) == before
