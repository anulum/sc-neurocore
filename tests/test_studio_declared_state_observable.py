# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Declared state must be observable on the instance

"""Gate for the join between a declared state table and the running model.

A Studio run records the state a model *declares*: the committed descriptor's
``[state]`` table, read off the constructed instance by name. The join is by
name and nothing checks it, so a model may declare ``ref_remaining`` and carry
``_ref_remaining``; the run then reports the declared variable as
non-observable, reports the private attribute as ``undeclared_mutable``, and
publishes a custody verdict of incomplete for a register the model tracks
exactly. Both halves of that failure are silent — the descriptor parses, the
model runs, mypy and ruff see nothing, and only the run's own custody payload
carries the reason.

This gate holds the join for the whole catalogue. Every declared variable must
be an attribute of a default-constructed instance, and the variables that are
attributes but that a run still cannot record are pinned by name, so a new one
is a test failure rather than a quiet demotion of somebody's run.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.studio.state_layout import (
    ObservedState,
    attribute_fingerprints,
    declared_state,
    observe_layout,
    undeclared_mutations,
)

# Declared variables that are attributes of the instance but that a run cannot
# record, with the exact reason the layout reports. `excited` is a Python bool;
# the layout admits real scalars and numeric arrays, and the schema lowers the
# same register as 0.0/1.0, so the value is representable but not admitted.
# Recorded as DISCOVERED-DECLARED-BOOL-STATE-NOT-RECORDABLE; this pin keeps the
# set from growing while that row is open.
UNRECORDABLE_DECLARED_STATE: dict[tuple[str, str], str] = {
    ("LapicqueNeuron", "excited"): "unsupported state value (bool)",
}

# Steps and drive for the two refractory regressions: enough steps at a drive
# that spikes, so the refractory register is entered, decremented and left
# non-zero rather than sitting at its initial value for the whole run.
REFRACTORY_STEPS = 400

# Compte-WM integrates at 0.02 ms with a 20 ms membrane time constant, so the
# drive has to carry the membrane 20 mV within 8 ms of run: 2.0 nA against the
# 0.025 uS leak asymptotes 80 mV above rest and crosses threshold well inside
# the window.
COMPTE_WM_DRIVE_NA = 2.0


def _instance(class_name: str) -> Any:
    """Return a default-constructed catalogue model."""
    module = importlib.import_module(f"sc_neurocore.neurons.models.{_CLASS_TO_MODULE[class_name]}")
    model: Any = getattr(module, class_name)()
    return model


def _observed(class_name: str, *, n_steps: int = 256) -> tuple[ObservedState, ...]:
    """Return the observed layout variables of a default-constructed model."""
    source, stem, declared = declared_state(class_name)
    instance = _instance(class_name)
    layout = observe_layout(
        instance, source, stem, declared, n_steps=n_steps, element_budget=1 << 20
    )
    return layout.variables


def test_every_declared_state_variable_is_an_attribute_of_the_instance() -> None:
    """No catalogue model may declare state under a name it does not carry."""
    missing = [
        f"{class_name}.{variable.name}"
        for class_name in sorted(_CLASS_TO_MODULE)
        for variable in _observed(class_name)
        if variable.reason == "not an attribute of the model instance"
    ]
    assert missing == []


def test_unrecordable_declared_state_is_exactly_the_pinned_set() -> None:
    """Declared state a run cannot record must be the recorded set, and no more."""
    unrecordable = {
        (class_name, variable.name): variable.reason
        for class_name in sorted(_CLASS_TO_MODULE)
        for variable in _observed(class_name)
        if not variable.observable
    }
    assert unrecordable == UNRECORDABLE_DECLARED_STATE


@pytest.mark.parametrize("class_name", ["BrunelWangNeuron", "CompteWMNeuron"])
def test_refractory_register_is_declared_observed_and_returned(class_name: str) -> None:
    """The refractory register carries one name across descriptor, instance and state."""
    _, _, declared = declared_state(class_name)
    assert "ref_remaining" in {variable.name for variable in declared}

    observed = {variable.name: variable for variable in _observed(class_name)}
    assert observed["ref_remaining"].observable
    assert observed["ref_remaining"].kind == "scalar"
    assert observed["ref_remaining"].reason == ""

    assert "ref_remaining" in _instance(class_name).get_state()


def test_brunel_wang_run_leaves_no_undeclared_mutation() -> None:
    """A Brunel-Wang run that spikes records its refractory register as declared state."""
    neuron = _instance("BrunelWangNeuron")
    source, stem, declared = declared_state("BrunelWangNeuron")
    layout = observe_layout(
        neuron, source, stem, declared, n_steps=REFRACTORY_STEPS, element_budget=1 << 20
    )
    before = attribute_fingerprints(neuron)
    spikes = 0
    entered_refractory = False
    for _ in range(REFRACTORY_STEPS):
        spikes += neuron.step(1.0, 0.5, 0.5, 0.0)
        entered_refractory = entered_refractory or neuron.ref_remaining > 0.0
    assert spikes > 0
    assert entered_refractory

    layout = layout.with_undeclared_mutable(
        undeclared_mutations(before, attribute_fingerprints(neuron), layout)
    )
    assert layout.incomplete_reasons() == ()
    assert layout.complete


def test_compte_wm_run_leaves_no_undeclared_mutation() -> None:
    """A Compte-WM run that spikes records its refractory register as declared state."""
    neuron = _instance("CompteWMNeuron")
    source, stem, declared = declared_state("CompteWMNeuron")
    layout = observe_layout(
        neuron, source, stem, declared, n_steps=REFRACTORY_STEPS, element_budget=1 << 20
    )
    before = attribute_fingerprints(neuron)
    spikes = 0
    entered_refractory = False
    for _ in range(REFRACTORY_STEPS):
        spikes += neuron.step(COMPTE_WM_DRIVE_NA)
        entered_refractory = entered_refractory or neuron.ref_remaining > 0.0
    assert spikes > 0
    assert entered_refractory

    layout = layout.with_undeclared_mutable(
        undeclared_mutations(before, attribute_fingerprints(neuron), layout)
    )
    assert layout.incomplete_reasons() == ()
    assert layout.complete


def test_refractory_register_rejects_a_negative_value_by_its_public_name() -> None:
    """The refractory guard names the attribute a caller can actually reach."""
    neuron = _instance("BrunelWangNeuron")
    neuron.ref_remaining = -1.0
    with pytest.raises(ValueError, match="ref_remaining must be finite and non-negative"):
        neuron.step(0.0, 0.0, 0.0, 0.0)
