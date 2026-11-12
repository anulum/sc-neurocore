# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Behaviour cases of a model descriptor: what parses, what is refused, what a merge keeps

"""Bind the behaviour cases a model descriptor may state.

A case is the author's statement of what the model does at a named drive: an
interval on one observable with both ends, or a strict order between two
drives with a least margin. A bound with one end, an order without a margin
and an observable the descriptor does not declare are refused. A spike count
of a run lies between zero and the number of steps, so bounds that span that
whole range, an end above it and a margin above it are refused as well: a case
that cannot fail, or cannot hold, cannot be written.
"""

from __future__ import annotations

import copy
import tomllib
from typing import Any

import pytest
import tomli_w

from sc_neurocore.neurons.descriptor_generator import merge_descriptor_payloads
from sc_neurocore.neurons.model_descriptor import ModelDescriptorError, parse_model_descriptor


def _payload(*cases: dict[str, Any]) -> dict[str, Any]:
    """Return a minimal descriptor payload of a two-input model with the given cases."""
    payload: dict[str, Any] = {
        "metadata": {
            "schema_version": 2,
            "name": "Two-compartment demonstration",
            "class_name": "DendriticNMDANeuron",
            "module": "dendritic_nmda",
        },
        "state": {"v_soma": {"init": -65.0}, "v_dend": {"init": -65.0}},
        "parameters": {"g_nmda": {"default": 1.5}},
        "integration": {"dt": 0.1},
    }
    if cases:
        payload["behaviour"] = [copy.deepcopy(case) for case in cases]
    return payload


_INTERVAL: dict[str, Any] = {
    "name": "both inputs together fire the soma",
    "kind": "interval",
    "steps": 20000,
    "observable": "spikes",
    "inputs": {"i_soma": 20.0, "glutamate": 1.0},
    "bounds": [50, 70],
}

_CONTRAST: dict[str, Any] = {
    "name": "glutamate adds spikes at a firing current",
    "kind": "contrast",
    "steps": 20000,
    "observable": "spikes",
    "inputs": {"i_soma": 40.0, "glutamate": 1.0},
    "reference_inputs": {"i_soma": 40.0, "glutamate": 0.0},
    "margin": 10,
}

_STATE: dict[str, Any] = {
    "name": "glutamate alone raises the dendrite to its balance point",
    "kind": "interval",
    "steps": 20000,
    "observable": "state:v_dend",
    "inputs": {"i_soma": 0.0, "glutamate": 1.0},
    "bounds": [-60.0, -59.0],
}


def _changed(case: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """Return a copy of a case with keys replaced, or removed when the value is ``...``."""
    changed = copy.deepcopy(case)
    for key, value in changes.items():
        if value is ...:
            changed.pop(key, None)
        else:
            changed[key] = value
    return changed


def test_a_descriptor_without_the_table_has_no_case() -> None:
    """The table is optional; every descriptor written before it stays valid."""
    assert parse_model_descriptor(_payload()).behaviour == ()


def test_an_interval_case_parses_with_both_ends() -> None:
    """Name, drive, steps, observable and the two bounds are kept as written."""
    (case,) = parse_model_descriptor(_payload(_INTERVAL)).behaviour
    assert case.name == "both inputs together fire the soma"
    assert case.kind == "interval"
    assert case.steps == 20000
    assert case.observable == "spikes"
    assert case.inputs == (("glutamate", 1.0), ("i_soma", 20.0))
    assert case.bounds == (50.0, 70.0)
    assert case.reference_inputs == ()
    assert case.margin is None


def test_an_exact_value_is_an_interval_with_equal_ends() -> None:
    """A count of exactly zero has two ends; it is not a one-sided bound."""
    (case,) = parse_model_descriptor(_payload(_changed(_INTERVAL, bounds=[0, 0]))).behaviour
    assert case.bounds == (0.0, 0.0)


def test_a_contrast_case_parses_with_its_reference_drive_and_margin() -> None:
    """The second drive and the least difference are part of the case."""
    (case,) = parse_model_descriptor(_payload(_CONTRAST)).behaviour
    assert case.kind == "contrast"
    assert case.inputs == (("glutamate", 1.0), ("i_soma", 40.0))
    assert case.reference_inputs == (("glutamate", 0.0), ("i_soma", 40.0))
    assert case.margin == 10.0
    assert case.bounds is None


def test_a_declared_state_variable_is_an_observable() -> None:
    """The final value of a state variable the descriptor declares can be bounded."""
    (case,) = parse_model_descriptor(_payload(_STATE)).behaviour
    assert case.observable == "state:v_dend"
    assert case.bounds == (-60.0, -59.0)


def test_cases_keep_their_order_and_exported_type() -> None:
    """Several cases come back in file order as the exported case type."""
    from sc_neurocore.neurons.model_descriptor import BehaviourCase

    cases = parse_model_descriptor(_payload(_INTERVAL, _CONTRAST, _STATE)).behaviour
    assert [case.kind for case in cases] == ["interval", "contrast", "interval"]
    assert all(isinstance(case, BehaviourCase) for case in cases)


_REFUSED: dict[str, tuple[Any, str]] = {
    "the table is not a list": ({"name": "x"}, "list of tables"),
    "an entry is not a table": (["spikes"], "list of tables"),
    "no name": ([_changed(_INTERVAL, name=...)], "name"),
    "an empty name": ([_changed(_INTERVAL, name="")], "name"),
    "two cases with one name": ([_INTERVAL, _INTERVAL], "more than once"),
    "an unknown kind": ([_changed(_INTERVAL, kind="at-least")], "kind"),
    "no steps": ([_changed(_INTERVAL, steps=...)], "steps must be a positive whole number"),
    "zero steps": ([_changed(_INTERVAL, steps=0)], "steps must be a positive whole number"),
    "a negative step count": (
        [_changed(_INTERVAL, steps=-5)],
        "steps must be a positive whole number",
    ),
    "a fractional step count": (
        [_changed(_INTERVAL, steps=1.5)],
        "steps must be a positive whole number",
    ),
    "a boolean step count": (
        [_changed(_INTERVAL, steps=True)],
        "steps must be a positive whole number",
    ),
    "an unknown observable": ([_changed(_INTERVAL, observable="energy")], "observable"),
    "a state variable the descriptor does not declare": (
        [_changed(_STATE, observable="state:calcium")],
        "observable",
    ),
    "no drive": ([_changed(_INTERVAL, inputs=...)], "inputs"),
    "an empty drive": ([_changed(_INTERVAL, inputs={})], "inputs"),
    "a drive that is not a table": ([_changed(_INTERVAL, inputs=[20.0, 1.0])], "inputs"),
    "a drive value that is not a number": (
        [_changed(_INTERVAL, inputs={"i_soma": "strong"})],
        "inputs",
    ),
    "a boolean drive value": ([_changed(_INTERVAL, inputs={"i_soma": True})], "inputs"),
    "a drive value that is not finite": (
        [_changed(_INTERVAL, inputs={"i_soma": float("inf")})],
        "inputs",
    ),
    "an interval without bounds": ([_changed(_INTERVAL, bounds=...)], "bounds"),
    "a bound with one end": ([_changed(_INTERVAL, bounds=[50])], "bounds"),
    "bounds in the wrong order": ([_changed(_INTERVAL, bounds=[70, 50])], "bounds"),
    "a bound that is not a number": ([_changed(_INTERVAL, bounds=[50, "many"])], "bounds"),
    "a bound that is not finite": ([_changed(_STATE, bounds=[-60.0, float("inf")])], "bounds"),
    "a fractional spike count": (
        [_changed(_INTERVAL, bounds=[50.5, 70])],
        "a spike count is a whole number",
    ),
    "a fractional upper spike count": (
        [_changed(_INTERVAL, bounds=[50, 70.5])],
        "a spike count is a whole number",
    ),
    "a negative spike count": (
        [_changed(_INTERVAL, bounds=[-1, 70])],
        "a spike count is a whole number",
    ),
    "an interval with a reference drive": (
        [_changed(_INTERVAL, reference_inputs={"i_soma": 20.0, "glutamate": 0.0})],
        "interval",
    ),
    "an interval with a margin": ([_changed(_INTERVAL, margin=5)], "interval"),
    "a contrast without a reference drive": (
        [_changed(_CONTRAST, reference_inputs=...)],
        "reference_inputs",
    ),
    "a contrast whose drives name different inputs": (
        [_changed(_CONTRAST, reference_inputs={"i_soma": 40.0})],
        "same inputs",
    ),
    "a contrast between two equal drives": (
        [_changed(_CONTRAST, reference_inputs={"i_soma": 40.0, "glutamate": 1.0})],
        "differ",
    ),
    "an order without a margin": ([_changed(_CONTRAST, margin=...)], "margin"),
    "a margin of zero": ([_changed(_CONTRAST, margin=0)], "margin"),
    "a negative margin": ([_changed(_CONTRAST, margin=-3)], "margin"),
    "a margin that is not finite": ([_changed(_CONTRAST, margin=float("inf"))], "margin"),
    "a contrast with bounds": ([_changed(_CONTRAST, bounds=[10, 30])], "contrast"),
    "a key the case does not have": ([_changed(_INTERVAL, tolerance=0.1)], "tolerance"),
    "spike bounds that every model satisfies": (
        [_changed(_INTERVAL, bounds=[0, 20000])],
        "hold for every model",
    ),
    "spike bounds wider than the run": (
        [_changed(_INTERVAL, bounds=[0, 1_000_000_000])],
        "cannot exceed the number of steps",
    ),
    "spike bounds no run can reach": (
        [_changed(_INTERVAL, bounds=[20001, 30000])],
        "cannot exceed the number of steps",
    ),
    "a spike bound whose upper end no run can reach": (
        [_changed(_INTERVAL, bounds=[50, 20001])],
        "cannot exceed the number of steps",
    ),
    "a spike margin no run can reach": (
        [_changed(_CONTRAST, margin=20001)],
        "a spike margin is a whole number",
    ),
    "a fractional spike margin": (
        [_changed(_CONTRAST, margin=0.5)],
        "a spike margin is a whole number",
    ),
    "a spike bound too large for a float": (
        [_changed(_INTERVAL, bounds=[0, 10**400])],
        "bounds must be two finite numbers",
    ),
    "a state bound too large for a float": (
        [_changed(_STATE, bounds=[-60.0, 10**400])],
        "bounds must be two finite numbers",
    ),
    "a margin too large for a float": (
        [_changed(_CONTRAST, margin=10**400)],
        "margin must be a positive finite number",
    ),
    "a drive value too large for a float": (
        [_changed(_INTERVAL, inputs={"i_soma": 10**400, "glutamate": 1.0})],
        "inputs.i_soma must be a finite number",
    ),
    "a reference drive value too large for a float": (
        [_changed(_CONTRAST, reference_inputs={"i_soma": -(10**400), "glutamate": 0.0})],
        "reference_inputs.i_soma must be a finite number",
    ),
}


@pytest.mark.parametrize("reason", sorted(_REFUSED))
def test_a_case_that_could_not_fail_or_cannot_be_read_is_refused(reason: str) -> None:
    """Each malformed table is refused with a message that names the fault."""
    table, fragment = _REFUSED[reason]
    payload = _payload()
    payload["behaviour"] = table
    with pytest.raises(ModelDescriptorError, match=fragment):
        parse_model_descriptor(payload)


_ONE_NATURAL_END: dict[str, tuple[dict[str, Any], tuple[float, float] | float]] = {
    "at most, down from one below the number of steps": (
        _changed(_INTERVAL, bounds=[0, 19999]),
        (0.0, 19999.0),
    ),
    "at least one spike": (_changed(_INTERVAL, bounds=[1, 20000]), (1.0, 20000.0)),
    "a spike at every step": (_changed(_INTERVAL, bounds=[20000, 20000]), (20000.0, 20000.0)),
    "a spike margin of the whole run": (_changed(_CONTRAST, margin=20000), 20000.0),
    "a spike margin of one": (_changed(_CONTRAST, margin=1), 1.0),
    "a fractional margin between two state values": (
        _changed(_CONTRAST, observable="state:v_dend", margin=0.5),
        0.5,
    ),
    "a state margin above the number of steps": (
        _changed(_CONTRAST, observable="state:v_dend", margin=30000),
        30000.0,
    ),
    "state bounds that the number of steps does not limit": (
        _changed(_STATE, bounds=[-100000.0, 100000.0]),
        (-100000.0, 100000.0),
    ),
}


@pytest.mark.parametrize("reason", sorted(_ONE_NATURAL_END))
def test_a_case_at_the_edge_of_what_a_run_can_show_parses(reason: str) -> None:
    """The limits of a spike count apply to spike counts only, and their edges can be stated."""
    case, expected = _ONE_NATURAL_END[reason]
    (parsed,) = parse_model_descriptor(_payload(case)).behaviour
    assert (parsed.bounds if parsed.kind == "interval" else parsed.margin) == expected


def test_a_refusal_names_the_case() -> None:
    """The message of a refusal inside the second case says which case it is."""
    with pytest.raises(ModelDescriptorError, match="glutamate adds spikes"):
        parse_model_descriptor(_payload(_INTERVAL, _changed(_CONTRAST, margin=0)))


def test_a_merge_keeps_the_cases_of_the_committed_descriptor() -> None:
    """A structural refresh of the corpus does not drop what the author stated."""
    curated = _payload(_INTERVAL, _CONTRAST)
    regenerated = _payload()
    merged = merge_descriptor_payloads(curated, regenerated)
    assert merged["behaviour"] == curated["behaviour"]
    assert merged["behaviour"] is not curated["behaviour"]
    assert len(parse_model_descriptor(merged).behaviour) == 2


def test_a_merge_adds_no_table_where_none_was_committed() -> None:
    """The generator states no behaviour by itself."""
    assert "behaviour" not in merge_descriptor_payloads(_payload(), _payload())


def test_the_cases_survive_the_generator_s_serialisation() -> None:
    """What the generator writes parses back to the same cases."""
    merged = merge_descriptor_payloads(_payload(_INTERVAL, _CONTRAST, _STATE), _payload())
    reread = tomllib.loads(tomli_w.dumps(merged))
    assert (
        parse_model_descriptor(reread).behaviour
        == parse_model_descriptor(_payload(_INTERVAL, _CONTRAST, _STATE)).behaviour
    )
