# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.benda_herz import BendaHerzNeuron
from sc_neurocore.neurons.models.sc_stochastic_rate_adaptation import (
    SCStochasticRateAdaptationNeuron,
)


def test_source_defaults_match_paper_example() -> None:
    neuron = BendaHerzNeuron()
    assert neuron.onset_gain == 60.0
    assert neuron.adaptation_slope == 0.1
    assert neuron.a == neuron.phase == 0.0


def test_square_root_onset_example() -> None:
    neuron = BendaHerzNeuron()
    assert neuron._f_onset(4.0) == pytest.approx(120.0)
    assert neuron._f_onset(-1.0) == 0.0


def test_equation_eight_adaptation_rhs() -> None:
    neuron = BendaHerzNeuron(a=2.0, adaptation_slope=0.1, tau_a=100.0)
    da, phase_rate = neuron._rhs(2.0, 6.0)
    rate = 60.0 * math.sqrt(4.0)
    assert da == pytest.approx((0.1 * rate - 2.0) / 100.0)
    assert phase_rate == pytest.approx(rate / 1000.0)


def test_phase_generator_is_deterministic_and_resets_exactly() -> None:
    left = BendaHerzNeuron(phase=0.99, dt=1.0, adaptation_slope=0.0)
    right = BendaHerzNeuron(phase=0.99, dt=1.0, adaptation_slope=0.0)
    assert left.step(1.0) == right.step(1.0) == 1
    assert left.phase == right.phase == 0.0


def test_candidate_is_atomic_on_invalid_phase_increment() -> None:
    neuron = BendaHerzNeuron(a=0.5, phase=0.5, onset_gain=1.0e6, dt=1.0)
    with pytest.raises(ValueError, match="at most one spike"):
        neuron.step(1.0e6)
    assert neuron.a == 0.5
    assert neuron.phase == 0.5


def test_source_and_project_identities_are_distinct() -> None:
    source = BendaHerzNeuron()
    project = SCStochasticRateAdaptationNeuron(seed=7)
    assert type(source) is not type(project)
    assert not hasattr(source, "seed")
    assert hasattr(project, "seed")


@pytest.mark.parametrize(
    ("field", "value"),
    [("a", -1.0), ("phase", 1.0), ("onset_gain", 0.0), ("tau_a", 0.0), ("dt", 0.0)],
)
def test_source_rejects_invalid_configuration(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        BendaHerzNeuron(**{field: value})
