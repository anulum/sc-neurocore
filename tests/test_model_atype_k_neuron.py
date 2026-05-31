import math

import pytest

from sc_neurocore.neurons.models.atype_k_neuron import ATypeKNeuron


def snapshot(neuron: ATypeKNeuron) -> tuple[float, float, float, float, float]:
    return neuron.v, neuron.h, neuron.n, neuron.a, neuron.b


def test_default_step_preserves_finite_gate_probabilities() -> None:
    neuron = ATypeKNeuron()

    spike = neuron.step(0.0)

    assert spike in (0, 1)
    assert math.isfinite(neuron.v)
    assert 0.0 <= neuron.h <= 1.0
    assert 0.0 <= neuron.n <= 1.0
    assert 0.0 <= neuron.a <= 1.0
    assert 0.0 <= neuron.b <= 1.0


def test_a_type_current_delays_depolarization_relative_to_no_ia_current() -> None:
    with_ia = ATypeKNeuron(g_na=0.0, g_k=0.0, g_a=8.0, g_l=0.0, a=1.0, b=1.0)
    without_ia = ATypeKNeuron(g_na=0.0, g_k=0.0, g_a=0.0, g_l=0.0, a=1.0, b=1.0)

    with_ia.step(20.0)
    without_ia.step(20.0)

    assert with_ia.v < without_ia.v


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0.0},
        {"c_m": 0.0},
        {"phi": 0.0},
        {"h": -0.1},
        {"n": 1.1},
        {"a": math.nan},
        {"b": math.inf},
        {"g_a": -1.0},
        {"gain": math.inf},
        {"_sub_steps": 0},
    ],
)
def test_invalid_physical_configuration_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ATypeKNeuron(**kwargs)


def test_non_finite_current_does_not_mutate_state() -> None:
    neuron = ATypeKNeuron()
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(math.nan)

    assert snapshot(neuron) == before


def test_corrupted_runtime_gate_does_not_mutate_state() -> None:
    neuron = ATypeKNeuron()
    neuron.a = 1.5
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(1.0)

    assert snapshot(neuron) == before


def test_unstable_input_drive_does_not_mutate_state() -> None:
    neuron = ATypeKNeuron()
    neuron.gain = 1e308
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(1e308)

    assert snapshot(neuron) == before


def test_candidate_outside_safety_bounds_does_not_mutate_state() -> None:
    neuron = ATypeKNeuron(g_na=0.0, g_k=0.0, g_a=0.0, g_l=0.0, c_m=1e-12, v_threshold=1e30)
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(1.0)

    assert snapshot(neuron) == before
