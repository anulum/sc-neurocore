# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC-NIR schema validation error branches

"""Contracts for SC-NIR payload validation, source/precision and online-learning edges."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest

from sc_neurocore.ir.scnir_schema import (
    SCNIR_SCHEMA_VERSION,
    SCNIR_V06_SCHEMA_VERSION,
    SCNIRCorrelationConstraint,
    SCNIRDocument,
    SCNIRPrecision,
    SCNIRSource,
    SCNIRStream,
    SCNIRStreamTransform,
    SCNIRValidationError,
    _infer_legacy_signal_kind,
    _is_prime,
    scnir_from_dict,
    scnir_to_dict,
    upgrade_scnir_dict,
    validate_scnir_dict,
)

_ANNOTATION: dict[str, Any] = {
    "schema_version": "sc-neurocore.online-o1.annotation.v1",
    "rule_id": "stdp0",
    "rule_family": "reward_modulated_stdp",
    "state_fields": ["weight", "pre_trace", "post_trace", "eligibility"],
    "per_synapse_state_bits": 20,
    "weight_bits": 8,
    "trace_bits": 4,
    "reward_bits": 4,
    "learning_shift": 0,
    "trace_decay_shift": 0,
    "saturation_policy": "signed_eligibility_unsigned_weight",
    "hidden_history_fields": [],
    "sequence_length_independent": True,
}


def _valid_document() -> SCNIRDocument:
    """A complete SC-NIR document with a transform and an online-learning annotation."""
    return SCNIRDocument(
        producer="sc-neurocore-test",
        streams=[
            SCNIRStream(
                stream_id="layer0_input",
                layer="layer0",
                bitstream_length=1024,
                encoding="bipolar",
                signal_kind="spike",
                precision=SCNIRPrecision(
                    signed=True,
                    total_bits=16,
                    fractional_bits=8,
                    accumulator_bits=32,
                    rounding="nearest_even",
                    overflow="saturate",
                ),
                source=SCNIRSource(
                    kind="lfsr",
                    seed=17,
                    lfsr_polynomial="x^16 + x^14 + x^13 + x^11 + 1",
                    tap_mask=0xB400,
                ),
                transforms=[
                    SCNIRStreamTransform(
                        kind="threshold",
                        position="source",
                        comparison="greater_than",
                        values=(0.5,),
                    )
                ],
                correlation_constraints=[
                    SCNIRCorrelationConstraint(
                        peer_stream_id="layer0_weight",
                        policy="max_correlation",
                        max_abs_correlation=0.03,
                    )
                ],
            ),
            SCNIRStream(
                stream_id="layer0_weight",
                layer="layer0",
                bitstream_length=1024,
                encoding="unipolar",
                signal_kind="weight",
                precision=SCNIRPrecision(
                    signed=False,
                    total_bits=12,
                    fractional_bits=10,
                    accumulator_bits=24,
                    rounding="stochastic",
                    overflow="error",
                ),
                source=SCNIRSource(kind="sobol", sobol_dimension=3),
                online_learning=dict(_ANNOTATION),
            ),
        ],
    )


def _valid_payload() -> dict[str, Any]:
    """A validated JSON-ready SC-NIR payload that round-trips cleanly."""
    return scnir_to_dict(_valid_document())


def test_valid_payload_round_trips_through_typed_document() -> None:
    """The reference payload validates and survives a typed round-trip (incl. transforms)."""
    payload = _valid_payload()
    validate_scnir_dict(payload)

    assert scnir_to_dict(scnir_from_dict(payload)) == payload


def _set_online(field: str, value: Any) -> Callable[[dict[str, Any]], None]:
    """Mutator that overrides one online-learning annotation field on the weight stream."""

    def mutate(payload: dict[str, Any]) -> None:
        payload["streams"][1]["online_learning"][field] = value

    return mutate


def _set_source(field: str, value: Any) -> Callable[[dict[str, Any]], None]:
    """Mutator that overrides one source field on the input stream."""

    def mutate(payload: dict[str, Any]) -> None:
        payload["streams"][0]["source"][field] = value

    return mutate


def _chain(*mutators: Callable[[dict[str, Any]], None]) -> Callable[[dict[str, Any]], None]:
    """Compose several mutators into a single mutator applied in order."""

    def mutate(payload: dict[str, Any]) -> None:
        for mutator in mutators:
            mutator(payload)

    return mutate


_MUTATORS: list[tuple[Callable[[dict[str, Any]], None], str]] = [
    (lambda p: p.__setitem__("schema_version", "x"), "schema_version must be"),
    (lambda p: p.__setitem__("streams", []), "at least one stream"),
    (lambda p: p.__setitem__("streams", 123), "streams must be an array"),
    (lambda p: p.__setitem__("streams", [123]), r"streams\[0\] must be an object"),
    (lambda p: p.__setitem__("producer", ""), "producer must be a non-empty string"),
    (lambda p: p["streams"][0].__setitem__("stream_id", "bad id"), "invalid identifier syntax"),
    (
        lambda p: p["streams"][0]["precision"].__setitem__("signed", 1),
        "signed must be a boolean",
    ),
    (
        lambda p: p["streams"][0]["precision"].__setitem__("accumulator_bits", 8),
        "accumulator_bits must be >= total_bits",
    ),
    (lambda p: _set_source("seed", (1 << 64))(p), "seed must fit in uint64"),
    (
        lambda p: _chain(_set_source("kind", "halton"), _set_source("halton_base", 8))(p),
        "halton_base must be prime",
    ),
    (
        lambda p: _chain(_set_source("kind", "hardware"), _set_source("hardware_id", ""))(p),
        "hardware_id must be a non-empty string",
    ),
    (
        lambda p: _chain(_set_source("kind", "replay"), _set_source("replay_uri", ""))(p),
        "replay_uri must be a non-empty string",
    ),
    (
        lambda p: p["streams"][0]["correlation_constraints"][0].__setitem__("seed_domain", ""),
        "seed_domain must be a non-empty string",
    ),
    (
        lambda p: p["streams"][0]["transforms"][0].__setitem__("values", ["x"]),
        "values\\[0\\] must be numeric",
    ),
    (
        lambda p: p["streams"][0]["transforms"][0].__setitem__("values", [math.inf]),
        "values\\[0\\] must be finite",
    ),
    (
        lambda p: p["streams"][0]["correlation_constraints"][0].__setitem__(
            "max_abs_correlation", "x"
        ),
        "max_abs_correlation must be numeric",
    ),
    (
        lambda p: p["streams"][0]["correlation_constraints"][0].__setitem__(
            "max_abs_correlation", 1.5
        ),
        "max_abs_correlation must be in",
    ),
    (
        lambda p: p["streams"][0]["correlation_constraints"][0].__setitem__(
            "max_abs_correlation", None
        ),
        "max_abs_correlation is required",
    ),
    (_set_online("schema_version", "x"), "schema_version is unsupported"),
    (_set_online("rule_family", "x"), "rule_family is unsupported"),
    (_set_online("state_fields", ["weight"]), "state_fields must match"),
    (_set_online("trace_bits", 1), "trace_bits must be >= 2"),
    (_set_online("per_synapse_state_bits", 99), "per_synapse_state_bits must equal"),
    (_set_online("saturation_policy", "x"), "saturation_policy is unsupported"),
    (_set_online("hidden_history_fields", ["x"]), "hidden_history_fields must be empty"),
    (_set_online("sequence_length_independent", False), "sequence_length_independent must be true"),
]


@pytest.mark.parametrize(("mutator", "match"), _MUTATORS)
def test_validate_rejects_malformed_payloads(
    mutator: Callable[[dict[str, Any]], None], match: str
) -> None:
    """Each schema invariant rejects its specific malformed field."""
    payload = _valid_payload()
    mutator(payload)

    with pytest.raises(SCNIRValidationError, match=match):
        validate_scnir_dict(payload)


def test_is_prime_classifies_small_integers() -> None:
    """_is_prime covers the below-two, even, odd-composite and prime branches."""
    assert _is_prime(1) is False
    assert _is_prime(2) is True
    assert _is_prime(3) is True
    assert _is_prime(4) is False
    assert _is_prime(7) is True
    assert _is_prime(9) is False
    assert _is_prime(11) is True
    assert _is_prime(25) is False


def test_infer_legacy_signal_kind_maps_state_suffix() -> None:
    """A state/value-suffixed stream id is inferred as an analogue-state stream."""
    assert _infer_legacy_signal_kind("layer0.state") == "analogue_state"
    assert _infer_legacy_signal_kind("layer0.value") == "analogue_state"
    assert _infer_legacy_signal_kind("layer0_weight") == "weight"
    assert _infer_legacy_signal_kind("layer0_input") == "spike"


def test_upgrade_inserts_default_online_learning_for_v06_streams() -> None:
    """Upgrading a v0.6 payload backfills the absent online-learning annotation as null."""
    payload = _valid_payload()
    payload["schema_version"] = SCNIR_V06_SCHEMA_VERSION
    for stream in payload["streams"]:
        stream.pop("online_learning", None)

    upgraded = upgrade_scnir_dict(payload)

    assert upgraded["schema_version"] == SCNIR_SCHEMA_VERSION
    assert all(stream["online_learning"] is None for stream in upgraded["streams"])
