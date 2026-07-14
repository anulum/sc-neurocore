# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reference-trace payload validation

"""Parser contract tests for neuron reference-trace corpus payloads."""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import cast

import pytest

from sc_neurocore.neurons.reference_traces import (
    REFERENCE_TRACE_SCHEMA_VERSION,
    reference_trace_spec_from_payload,
)

PayloadMutation = Callable[[dict[str, object]], None]


def _valid_payload() -> dict[str, object]:
    return {
        "schema_version": REFERENCE_TRACE_SCHEMA_VERSION,
        "name": "candidate_lif_trace",
        "model": {
            "schema_name": "lif",
            "runner": "universal_dsl",
        },
        "protocol": {
            "dt": 0.1,
            "steps": 2,
            "inputs": {"I": 1.0},
            "state_variables": ["v"],
        },
        "provenance": {
            "kind": "analytic_closed_form",
            "source": "unit-test payload",
            "equation": "closed form",
            "citation": "candidate reference",
        },
        "expected_features": {
            "spike_count": 0.0,
        },
        "tolerances": {
            "default": {
                "absolute": 0.0,
                "relative": 0.0,
            }
        },
    }


def _section(payload: dict[str, object], key: str) -> dict[str, object]:
    return cast(dict[str, object], payload[key])


def test_payload_without_optional_citation_remains_valid() -> None:
    """Corpus authors may omit external citations for internal analytic seeds."""
    payload = _valid_payload()
    del _section(payload, "provenance")["citation"]

    spec = reference_trace_spec_from_payload(payload)

    assert spec.provenance.citation is None


def test_payload_accepts_a_genuinely_stateless_event_trace() -> None:
    """An empty state list records event features without inventing cell state."""
    payload = _valid_payload()
    _section(payload, "protocol")["state_variables"] = []

    spec = reference_trace_spec_from_payload(payload)

    assert spec.protocol.state_variables == ()


def test_payload_validation_rejects_malformed_corpus_contracts() -> None:
    """The public payload parser must fail closed on malformed corpus entries."""
    cases: list[tuple[PayloadMutation, str]] = [
        (
            lambda payload: payload.__setitem__("schema_version", "wrong"),
            "schema_version",
        ),
        (
            lambda payload: _section(payload, "model").__setitem__("runner", "python"),
            "runner",
        ),
        (
            lambda payload: _section(payload, "model").__setitem__("schema_name", "missing"),
            "not bundled",
        ),
        (
            lambda payload: payload.__delitem__("name"),
            "missing 'name'",
        ),
        (
            lambda payload: payload.__setitem__("name", ""),
            "non-empty string",
        ),
        (
            lambda payload: payload.__setitem__("model", "bad"),
            "must be an object",
        ),
        (
            lambda payload: payload.__setitem__("model", cast(dict[str, object], {1: "bad"})),
            "string keys",
        ),
        (
            lambda payload: _section(payload, "protocol").__setitem__("dt", True),
            "must be numeric",
        ),
        (
            lambda payload: _section(payload, "protocol").__setitem__("dt", math.inf),
            "must be finite",
        ),
        (
            lambda payload: _section(payload, "protocol").__setitem__("dt", 0.0),
            "must be positive",
        ),
        (
            lambda payload: _section(payload, "protocol").__setitem__("steps", True),
            "must be an integer",
        ),
        (
            lambda payload: _section(payload, "protocol").__setitem__("steps", 0),
            "must be positive",
        ),
        (
            lambda payload: _section(payload, "protocol").__setitem__("inputs", {}),
            "must not be empty",
        ),
        (
            lambda payload: payload.__setitem__("expected_features", {"spike_count": "bad"}),
            "expected_features.*spike_count",
        ),
        (
            lambda payload: payload.__setitem__("expected_features", {"spike_count": math.inf}),
            "must be finite",
        ),
        (
            lambda payload: _section(payload, "protocol").__setitem__("state_variables", "v"),
            "must be a list",
        ),
        (
            lambda payload: _section(payload, "protocol").__setitem__("state_variables", [""]),
            "non-empty strings",
        ),
        (
            lambda payload: _section(payload, "provenance").__setitem__("citation", ""),
            "non-empty string",
        ),
        (
            lambda payload: payload.__setitem__("tolerances", {}),
            "tolerance missing",
        ),
        (
            lambda payload: payload.__setitem__("tolerances", {"default": "bad"}),
            "tolerances.default",
        ),
        (
            lambda payload: payload.__setitem__(
                "tolerances", {"default": cast(dict[str, object], {1: "bad"})}
            ),
            "string keys",
        ),
        (
            lambda payload: payload.__setitem__("tolerances", {"default": {"absolute": -1.0}}),
            "non-negative",
        ),
        (
            lambda payload: payload.__setitem__("tolerances", {"default": {"absolute": math.inf}}),
            "must be finite",
        ),
        (
            lambda payload: payload.__setitem__(
                "tolerances", {"default": {"absolute": 0.0, "relative": "bad"}}
            ),
            "relative value must be numeric",
        ),
        (
            lambda payload: payload.__setitem__(
                "tolerances", {"default": {"absolute": 0.0, "relative": math.inf}}
            ),
            "relative value must be finite",
        ),
    ]

    for mutate, message in cases:
        payload = _valid_payload()
        mutate(payload)
        with pytest.raises(ValueError, match=message):
            reference_trace_spec_from_payload(payload)
