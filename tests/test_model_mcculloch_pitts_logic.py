# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — McCulloch-Pitts source logic contracts

"""Source truth tables, inhibition, and statelessness contracts."""

from .model_mcculloch_pitts_support import *


def test_defaults_are_a_stateless_positive_count_contract() -> None:
    """The default is one excitatory afferent and no invented membrane state."""
    neuron = McCullochPittsNeuron()
    assert dataclasses.is_dataclass(neuron)
    assert neuron.theta == 1
    assert type(neuron.theta) is int
    assert not hasattr(neuron, "v")


@pytest.mark.parametrize(
    ("count", "expected"),
    ((0, 0), (1, 1), (2, 1)),
)
def test_or_gate_is_the_theta_one_truth_table(count: int, expected: int) -> None:
    """One or more active excitatory afferents implement inclusive OR."""
    assert McCullochPittsNeuron(theta=1).step(count) == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    ((0, 0), (1, 0), (2, 1)),
)
def test_and_gate_is_the_theta_two_truth_table(count: int, expected: int) -> None:
    """Both of two excitatory afferents are required at theta two."""
    assert McCullochPittsNeuron(theta=2).step(count) == expected


def test_absolute_inhibition_vetoes_every_excitatory_count() -> None:
    """One inhibitory afferent dominates even an int32-maximum excitation."""
    neuron = McCullochPittsNeuron(theta=1)
    assert neuron.step(0, True) == 0
    assert neuron.step(1, True) == 0
    assert neuron.step(_INT32_MAX, True) == 0


def test_conjoined_negation_uses_the_source_inhibitory_wire() -> None:
    """The 1943 not-B conjunction is an absolute veto, not a negative weight."""
    neuron = McCullochPittsNeuron(theta=1)
    truth = {
        (0, False): 0,
        (1, False): 1,
        (0, True): 0,
        (1, True): 0,
    }
    assert {key: neuron.step(*key) for key in truth} == truth


def test_three_input_majority_uses_a_fixed_excitatory_count() -> None:
    """Theta two implements the majority of three binary excitatory afferents."""
    neuron = McCullochPittsNeuron(theta=2)
    assert [neuron.step(count) for count in range(4)] == [0, 0, 1, 1]


def test_calls_are_stateless_and_reset_is_a_validating_noop() -> None:
    """History cannot affect one-delay logical activity."""
    neuron = McCullochPittsNeuron(theta=2)
    assert [neuron.step(value) for value in (2, 0, 2, 0)] == [1, 0, 1, 0]
    neuron.reset()
    assert neuron.theta == 2
