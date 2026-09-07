# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A model that holds no state can say so

"""A model that genuinely evolves nothing declares emptiness, and is held to it.

`declared_state` answered `undeclared` in two different situations: a model
whose state nobody has declared, and a model that has no state to declare. The
second is a correct model, but it was reported with `complete=false` for ever
and an operator could not tell it from the first. Emptiness was a silence.

A descriptor now asserts `stateless` per identity. It is never inferred: no
static rule and no single drive can tell "holds nothing" from "was not
exercised", which is the whole reason the neighbouring private-register row is
blocked. What makes the assertion safe is that a run still audits it — an
asserted-stateless model that mutates anything is reported incomplete, naming
what moved — and that the two identities carrying it were measured over an
exercising drive first.

One candidate was refused. `InhomogeneousPoissonNeuron` mutates no attribute
either, but draws from the process-wide NumPy generator, so its run cannot be
reproduced from anything recorded. Declaring it complete would publish an
irreproducible run as fully recorded, and the case below holds that refusal to
the measurement that justifies it.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from sc_neurocore.neurons.descriptor_generator import generate_descriptor_payload
from sc_neurocore.neurons.model_catalogue import load_descriptor
from sc_neurocore.neurons.model_descriptor import ModelDescriptorError, parse_model_descriptor
from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.studio.model_simulate import simulate_model
from tests.engine_requirement import require_engine
from sc_neurocore.studio.state_layout import (
    DeclaredState,
    ObservedState,
    StateLayout,
    declared_state,
)

#: The identities that assert they hold no state, with a drive that exercises
#: each one. Both were measured over these steps before the assertion was made.
STATELESS: dict[str, list[tuple[object, ...]]] = {
    "McCullochPittsNeuron": [(3, False), (0, False), (5, True), (1, False)] * 25,
    "SiegertTransferFunction": [(1.0,), (20.0,), (-5.0,), (50.0,)] * 25,
}

#: Mutates no attribute either, and is deliberately NOT asserted: its output
#: comes from the process-wide generator, so a run of it is not reproducible
#: from anything the run records.
REFUSED = "InhomogeneousPoissonNeuron"
REFUSED_DRIVE: list[tuple[object, ...]] = [(500.0,), (0.0,), (900.0,)] * 40


def _model(class_name: str) -> Any:
    """Return a catalogue model class."""
    module = importlib.import_module(f"sc_neurocore.neurons.models.{_CLASS_TO_MODULE[class_name]}")
    return getattr(module, class_name)


def _attributes(instance: object) -> dict[str, str]:
    """Return a comparable fingerprint of every instance attribute."""
    names = set(vars(instance)) if hasattr(instance, "__dict__") else set()
    for klass in type(instance).__mro__:
        names |= set(getattr(klass, "__slots__", ()) or ())
    fingerprint: dict[str, str] = {}
    for name in sorted(names):
        value = getattr(instance, name, None)
        fingerprint[name] = (
            np.asarray(value).tobytes().hex() if isinstance(value, np.ndarray) else repr(value)
        )
    return fingerprint


#: Seed for the process-wide generator probe below.
PROBE_SEED = 20260908


def _run(
    class_name: str, drives: list[tuple[object, ...]]
) -> tuple[dict[str, str], dict[str, str], bool, bool]:
    """Drive a model and report what moved, on the instance and globally.

    The process-wide generator is probed by draw rather than by reading its
    internal state: the first draw after a fresh seed is recorded, the seed is
    set again, the model is driven, and one more draw is taken. A model that
    consumed nothing leaves that draw equal to the recorded one.
    """
    np.random.seed(PROBE_SEED)
    untouched = float(np.random.random())

    instance = _model(class_name)()
    before = _attributes(instance)
    np.random.seed(PROBE_SEED)
    outputs = [instance.step(*drive) for drive in drives]
    generator_moved = float(np.random.random()) != untouched
    return before, _attributes(instance), generator_moved, len({str(o) for o in outputs}) > 1


class TestTheAssertionIsMeasured:
    @pytest.mark.parametrize("class_name", sorted(STATELESS))
    def test_the_model_mutates_nothing_under_a_drive_that_exercises_it(
        self, class_name: str
    ) -> None:
        """The assertion is only as good as the run it was measured over."""
        before, after, _generator_moved, varied = _run(class_name, STATELESS[class_name])
        assert varied, "a drive that never changes the output has not exercised the model"
        assert before == after

    @pytest.mark.parametrize("class_name", sorted(STATELESS))
    def test_the_model_leaves_the_process_wide_generator_alone(self, class_name: str) -> None:
        """State held outside the instance is state the run cannot reproduce."""
        _before, _after, generator_moved, _varied = _run(class_name, STATELESS[class_name])
        assert generator_moved is False


class TestTheDescriptorCarriesTheAssertion:
    @pytest.mark.parametrize("class_name", sorted(STATELESS))
    def test_the_committed_descriptor_asserts_it(self, class_name: str) -> None:
        """What the layout reads is the committed file, so check the file."""
        descriptor = load_descriptor(class_name)
        assert descriptor is not None
        assert descriptor.stateless is True
        assert descriptor.state == ()

    @pytest.mark.parametrize("class_name", sorted(STATELESS))
    def test_the_generator_produces_it_without_the_committed_file(self, class_name: str) -> None:
        """A hand-edited descriptor is overwritten by the next regeneration."""
        assert generate_descriptor_payload(class_name)["metadata"]["stateless"] is True

    def test_the_set_that_asserts_it_is_the_measured_set(self) -> None:
        """A new assertion must arrive with its measurement, not on its own."""
        asserting = sorted(
            name
            for name in _CLASS_TO_MODULE
            if (descriptor := load_descriptor(name)) is not None and descriptor.stateless
        )
        assert asserting == sorted(STATELESS)

    def test_a_descriptor_cannot_assert_emptiness_and_declare_state(self) -> None:
        """The two claims contradict; the parser must not carry both."""
        payload = dict(generate_descriptor_payload("AmariNeuralField"))
        payload["metadata"] = {**payload["metadata"], "stateless": True}
        with pytest.raises(ModelDescriptorError, match="cannot be both"):
            parse_model_descriptor(payload)


class TestTheLayoutSeparatesSilenceFromEmptiness:
    @pytest.mark.parametrize("class_name", sorted(STATELESS))
    def test_an_asserted_model_answers_descriptor_with_no_variables(self, class_name: str) -> None:
        """The defect: this used to be indistinguishable from an undeclared model."""
        source, _profile, variables = declared_state(class_name)
        assert source == "descriptor"
        assert variables == ()

    def test_a_model_that_declares_nothing_and_asserts_nothing_stays_undeclared(self) -> None:
        """Emptiness must be a declaration, never the absence of one."""
        source, _profile, variables = declared_state("AstrocyteNeuron")
        assert source == "undeclared"
        assert variables == ()

    def test_the_source_stays_inside_the_published_union(self) -> None:
        """`studio.state-layout.v1` names three sources; this adds no fourth."""
        sources = {declared_state(name)[0] for name in _CLASS_TO_MODULE}
        assert sources <= {"descriptor", "undeclared"}


class TestTheRunReportsCompleteCustody:
    @pytest.mark.parametrize("class_name", sorted(STATELESS))
    def test_a_run_of_an_asserted_model_is_complete(self, class_name: str) -> None:
        """Recording everything a model declares is complete custody, even of nothing.

        Measured on the Python path, because that is the one this unit speaks
        for. The Rust batch lane adds custody limits of its own, held below.
        """
        layout = simulate_model(class_name, duration=20.0, current=5.0, use_fast_path=False)[
            "state_layout"
        ]
        assert layout["source"] == "descriptor"
        assert layout["complete"] is True
        assert layout["incomplete_reasons"] == []

    def test_the_rust_batch_lane_still_limits_the_custody_it_can_offer(self) -> None:
        """Declaring emptiness must not make a lane's own limitation disappear.

        Dispatched to the Rust batch backend, McCullochPitts is still reported
        incomplete — the lane exports a membrane voltage this logical neuron
        does not have, and exposes no initial snapshot. Both reasons name the
        backend rather than the model, which is DISCOVERED-RUST-BATCH-VOLTAGE-ONLY
        and not this row. The spikes agree across the two paths; only the
        custody claim differs.
        """
        require_engine()
        fast = simulate_model(
            "McCullochPittsNeuron", duration=20.0, current=5.0, use_fast_path=True
        )
        layout = fast["state_layout"]
        assert layout["source"] == "descriptor"
        assert layout["complete"] is False
        assert all("backend" in reason for reason in layout["incomplete_reasons"])
        slow = simulate_model(
            "McCullochPittsNeuron", duration=20.0, current=5.0, use_fast_path=False
        )
        assert fast["spikes"] == slow["spikes"]

    def test_the_refused_candidate_is_still_reported_incomplete(self) -> None:
        """It mutates nothing and is still not reproducible; custody must say so."""
        layout = simulate_model(REFUSED, duration=20.0, current=5.0)["state_layout"]
        assert layout["source"] == "undeclared"
        assert layout["complete"] is False

    def test_the_refusal_rests_on_a_measurement(self) -> None:
        """Why it is refused: no attribute moves, but the global generator does."""
        before, after, generator_moved, varied = _run(REFUSED, REFUSED_DRIVE)
        assert before == after
        assert varied
        assert generator_moved is True


class TestTheAssertionIsFalsifiable:
    def test_an_asserted_model_that_mutates_is_reported_incomplete(self) -> None:
        """This is what makes a per-identity assertion safe to accept."""
        layout = StateLayout(source="descriptor", schema_profile="probe", variables=())
        assert layout.complete is True

        audited = layout.with_undeclared_mutable(["_hidden"])
        assert audited.complete is False
        assert audited.incomplete_reasons() == (
            "_hidden: changed during the run but is not declared state",
        )

    def test_a_declared_variable_that_cannot_be_recorded_still_fails(self) -> None:
        """The assertion changes nothing for a model that does declare state."""
        spec = DeclaredState(name="v", role="unassigned", unit="", meaning="", declared_init=None)
        layout = StateLayout(
            source="descriptor",
            schema_profile="probe",
            variables=(ObservedState(spec, None, None, False, "not an attribute", "none"),),
        )
        assert layout.complete is False
