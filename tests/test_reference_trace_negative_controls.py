# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reference-trace negative controls

"""A corpus that only ever passes has not been shown to work.

These cases break the model on purpose — flipped drive sign, a unit scale off by
three orders of magnitude, the drive removed, events recorded on the wrong side
of the state update, a displaced threshold — and require the corpus to notice.
Where a control cannot bite, the reason is recorded rather than counted as a
pass, and the traces that stay blind are listed here so a new blindness has to
be added deliberately.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
import warnings

import pytest

from sc_neurocore.neurons.reference_trace_contracts import (
    ReferenceTraceProtocol,
    ReferenceTraceSpec,
)
from sc_neurocore.neurons.reference_trace_io import (
    list_reference_trace_specs,
    load_reference_trace_spec,
)
from sc_neurocore.neurons.reference_trace_mutations import (
    MUTATIONS,
    REFERENCE_TRACE_CONTROL_VERSION,
    negative_control_report,
    run_negative_control,
)

#: Applied controls the corpus does not catch, with the reason each is honest.
#:
#: * A logical threshold unit is scale-invariant above threshold, and its truth
#:   table does not distinguish pre-update from post-update state.
#: * A conductance spike that overshoots by tens of millivolts within one
#:   timestep crosses a displaced threshold on the same step, so these traces
#:   validate the trajectory rather than the threshold surface.
KNOWN_UNDETECTED = (
    ("connor_stevens_driven_spiking_doi", "threshold_shift"),
    ("exp_if_driven_rk4_doi", "threshold_shift"),
    ("hodgkin_huxley_driven_spiking_doi", "threshold_shift"),
    ("mcculloch_pitts_1943_truth_table", "event_ordering"),
    ("mcculloch_pitts_1943_truth_table", "unit_scale"),
    ("wang_buzsaki_driven_spiking_doi", "threshold_shift"),
)


@pytest.fixture(scope="module")
def report():
    """Run every control against every deterministic trace once."""
    return negative_control_report()


class TestSingleControls:
    def test_flipping_the_drive_sign_is_caught(self) -> None:
        outcome = run_negative_control(
            load_reference_trace_spec("hodgkin_huxley_driven_spiking_doi"), "sign"
        )

        assert outcome.status == "detected"
        assert outcome.mismatched_features > 0
        assert "sign" in outcome.reason

    def test_a_thousandfold_unit_error_is_caught(self) -> None:
        outcome = run_negative_control(
            load_reference_trace_spec("hodgkin_huxley_driven_spiking_doi"), "unit_scale"
        )

        assert outcome.caught

    def test_removing_the_drive_is_caught(self) -> None:
        outcome = run_negative_control(
            load_reference_trace_spec("hodgkin_huxley_driven_spiking_doi"), "omitted_current"
        )

        assert outcome.status == "detected"

    def test_recording_events_on_the_wrong_side_of_the_update_is_caught(self) -> None:
        outcome = run_negative_control(
            load_reference_trace_spec("adex_resting_adaptation_doi"), "event_ordering"
        )

        assert outcome.status == "detected"
        assert outcome.mismatched_features > 0

    def test_a_displaced_threshold_is_caught_where_the_trace_resolves_it(self) -> None:
        outcome = run_negative_control(
            load_reference_trace_spec("izhikevich2007_regular_spiking_doi"), "threshold_shift"
        )

        assert outcome.status == "detected"
        assert "displaced" in outcome.reason

    def test_a_zero_drive_protocol_reports_the_control_as_inapplicable(self) -> None:
        """Negating nothing is not a passed control."""
        outcome = run_negative_control(
            load_reference_trace_spec("adex_resting_adaptation_doi"), "sign"
        )

        assert outcome.status == "inapplicable"
        assert outcome.caught is False
        assert "zero input" in outcome.reason

    def test_a_silent_trace_cannot_control_its_threshold(self) -> None:
        outcome = run_negative_control(
            load_reference_trace_spec("quadratic_if_zero_current_analytic"), "threshold_shift"
        )

        assert outcome.status == "inapplicable"
        assert "no event" in outcome.reason

    def test_an_unknown_control_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown negative control"):
            run_negative_control(
                load_reference_trace_spec("adex_resting_adaptation_doi"),
                "wishful_thinking",  # type: ignore[arg-type]
            )

    def test_a_control_never_edits_the_committed_specification(self) -> None:
        spec = load_reference_trace_spec("hodgkin_huxley_driven_spiking_doi")
        before = dict(spec.protocol.inputs)

        for mutation in MUTATIONS:
            run_negative_control(spec, mutation)

        assert dict(spec.protocol.inputs) == before
        assert dict(load_reference_trace_spec(spec.name).protocol.inputs) == before


def _with_protocol(spec: ReferenceTraceSpec, **changes: object) -> ReferenceTraceSpec:
    """Return the spec with one protocol field replaced."""
    fields: dict[str, object] = {
        "dt": spec.protocol.dt,
        "steps": spec.protocol.steps,
        "inputs": spec.protocol.inputs,
        "state_variables": spec.protocol.state_variables,
        "parameter_overrides": spec.protocol.parameter_overrides,
    }
    fields.update(changes)
    return replace(spec, protocol=ReferenceTraceProtocol(**fields))  # type: ignore[arg-type]


class TestThresholdResolution:
    def test_an_overridden_threshold_is_displaced_from_the_operating_profile(self) -> None:
        """A profile that moves the threshold is what the control must displace.

        Reading the schema default instead would displace a value the protocol
        never used, and the control would measure the wrong surface.
        """
        spec = load_reference_trace_spec("izhikevich2007_regular_spiking_doi")
        profiled = _with_protocol(spec, parameter_overrides=MappingProxyType({"vpeak": 20.0}))

        outcome = run_negative_control(profiled, "threshold_shift")

        assert outcome.status == "detected"
        assert "from 20" in outcome.reason

    def test_an_ambiguous_threshold_condition_is_not_guessed(self) -> None:
        """Two candidate parameters means the control cannot say which is the threshold."""
        spec = load_reference_trace_spec("ermentrout_kopell_theta_euler_doi")

        outcome = run_negative_control(spec, "threshold_shift")

        assert outcome.status == "inapplicable"
        assert "no single adjustable parameter" in outcome.reason

    def test_a_malformed_schema_threshold_is_reported_not_guessed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A threshold this build cannot read is a boundary, not a passed control."""
        import sc_neurocore.neurons.reference_trace_mutations as mutations

        class _OpaqueSchema:
            def to_json(self) -> str:
                return '{"threshold": {"condition": 17}, "parameters": {"v_threshold": -50.0}}'

        monkeypatch.setattr(
            mutations.UniversalNeuron, "from_schema", classmethod(lambda *a, **k: _OpaqueSchema())
        )
        spec = load_reference_trace_spec("izhikevich2007_regular_spiking_doi")

        outcome = run_negative_control(spec, "threshold_shift")

        assert outcome.status == "inapplicable"
        assert "no single adjustable parameter" in outcome.reason

    def test_a_model_without_a_threshold_has_no_surface_to_displace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rate model emits no event by crossing anything."""
        import sc_neurocore.neurons.reference_trace_mutations as mutations

        class _NoThreshold:
            def to_json(self) -> str:
                return '{"parameters": {"tau": 10.0}}'

        monkeypatch.setattr(
            mutations.UniversalNeuron, "from_schema", classmethod(lambda *a, **k: _NoThreshold())
        )
        spec = load_reference_trace_spec("izhikevich2007_regular_spiking_doi")

        outcome = run_negative_control(spec, "threshold_shift")

        assert outcome.status == "inapplicable"

    def test_a_non_numeric_threshold_cannot_be_displaced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sc_neurocore.neurons.reference_trace_mutations as mutations

        class _TextThreshold:
            def to_json(self) -> str:
                return (
                    '{"threshold": {"condition": "v >= v_threshold"},'
                    ' "parameters": {"v_threshold": "high"}}'
                )

        monkeypatch.setattr(
            mutations.UniversalNeuron, "from_schema", classmethod(lambda *a, **k: _TextThreshold())
        )
        spec = load_reference_trace_spec("izhikevich2007_regular_spiking_doi")

        outcome = run_negative_control(spec, "threshold_shift")

        assert outcome.status == "inapplicable"

    def test_a_reordered_run_that_cannot_complete_is_a_refusal(self) -> None:
        """A protocol naming a state variable the model has no such thing as."""
        spec = load_reference_trace_spec("izhikevich2007_regular_spiking_doi")
        broken = _with_protocol(spec, state_variables=("no_such_variable",))

        outcome = run_negative_control(broken, "event_ordering")

        assert outcome.status == "refused"
        assert "could not complete" in outcome.reason


class TestCorpusControls:
    def test_no_trace_is_left_uncontrolled(self, report) -> None:
        """Every trace must fail under at least one broken model."""
        assert report.uncontrolled_traces() == ()

    def test_every_trace_and_control_has_an_outcome(self, report) -> None:
        expected = len(list_reference_trace_specs()) * len(MUTATIONS)

        assert report.version == REFERENCE_TRACE_CONTROL_VERSION
        assert len(report.outcomes) == expected
        assert {outcome.mutation for outcome in report.outcomes} == set(MUTATIONS)

    def test_the_blind_spots_are_exactly_the_ones_recorded(self, report) -> None:
        """A new undetected control is a claim change and must be added here."""
        undetected = tuple(
            sorted((outcome.name, outcome.mutation) for outcome in report.with_status("undetected"))
        )

        assert undetected == KNOWN_UNDETECTED

    def test_an_inapplicable_control_always_states_why(self, report) -> None:
        for outcome in report.with_status("inapplicable"):
            assert outcome.reason
            assert outcome.mismatched_features == 0
            assert outcome.caught is False

    def test_every_detection_names_at_least_one_violated_feature(self, report) -> None:
        for outcome in report.with_status("detected"):
            assert outcome.mismatched_features > 0

    def test_the_controls_do_not_leak_numeric_warnings(self) -> None:
        """A mutant leaving its numeric domain is expected, not a warning to raise."""
        spec = load_reference_trace_spec("hodgkin_huxley_driven_spiking_doi")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            outcome = run_negative_control(spec, "unit_scale")

        assert outcome.caught

    def test_the_report_is_json_safe(self, report) -> None:
        payload = report.to_public_dict()

        assert payload["version"] == REFERENCE_TRACE_CONTROL_VERSION
        assert payload["uncontrolled_traces"] == []
        outcomes = payload["outcomes"]
        assert isinstance(outcomes, list)
        assert len(outcomes) == len(report.outcomes)
        assert all(
            set(row) == {"mismatched_features", "mutation", "name", "reason", "status"}
            for row in outcomes
        )
