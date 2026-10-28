# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Philox stochastic-threshold runtime and seed-alias declaration edges

"""The Philox threshold runtime reports its key and keeps its counter on a refused step."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.equation_refusals import EquationStateFailure
from sc_neurocore.neurons.seed_domain import seed_aliases

_SEED = 2026


def _philox_poisson(probability: float) -> EquationNeuron:
    """Return a stateless Poisson neuron with a Philox threshold."""
    return EquationNeuron(
        equations={"v": "0.0"},
        state={"v": 0.0},
        parameters={"p": probability},
        dt=1.0,
        method="map",
        detection="poisson",
        probability_expression="p",
        rng_seed=_SEED,
        random_stream=3,
        threshold_rng="philox",
    )


def test_philox_threshold_reports_its_key() -> None:
    """The Philox stochastic-threshold runtime reports the neuron's seed as its initial seed."""
    neuron = _philox_poisson(0.5)
    assert neuron.stochastic_rng_initial_seed == _SEED


def test_a_refused_step_returns_the_philox_counter() -> None:
    """A probability outside [0, 1] is refused and the trial counter is left where it was."""
    neuron = _philox_poisson(0.5)
    neuron.step(0.0)
    neuron.step(0.0)
    before = neuron.stochastic_rng_state
    neuron.parameters["p"] = 1.5
    with pytest.raises(EquationStateFailure, match="stochastic spike probability"):
        neuron.step(0.0)
    assert neuron.stochastic_rng_state == before


class _IntegerAliases:
    SEED_ALIASES = {0: 0xACE1, 7: 9}


class _StringTarget:
    SEED_ALIASES = {0: "0xACE1"}


class _BoolRequest:
    SEED_ALIASES = {True: 5}


def test_seed_aliases_keep_an_integer_mapping() -> None:
    """A well-formed declaration is returned as declared."""
    assert seed_aliases(_IntegerAliases) == {0: 0xACE1, 7: 9}


@pytest.mark.parametrize("model", [_StringTarget, _BoolRequest])
def test_seed_aliases_ignore_a_declaration_that_is_not_integer_to_integer(model: type) -> None:
    """A non-integer or boolean seed makes the whole declaration void."""
    assert seed_aliases(model) == {}
