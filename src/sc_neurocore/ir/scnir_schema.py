# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC-NIR schema and fail-closed validator

"""SC-aware NIR metadata schema and validator.

The SC-NIR layer records stochastic-computing semantics that plain NIR does
not carry: stream length, encoding, fixed-point precision, random-source
metadata, deterministic stream transforms, and correlation constraints.
Validation is intentionally fail-closed so unrecognised or under-specified
metadata cannot silently reach hardware generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence, cast

SCNIR_SCHEMA_VERSION = "sc-neurocore.scnir.v0.7"
SCNIR_V06_SCHEMA_VERSION = "sc-neurocore.scnir.v0.6"
SCNIR_PREVIOUS_SCHEMA_VERSION = "sc-neurocore.scnir.v0.5"
SCNIR_V04_SCHEMA_VERSION = "sc-neurocore.scnir.v0.4"
SCNIR_V03_SCHEMA_VERSION = "sc-neurocore.scnir.v0.3"
SCNIR_V02_SCHEMA_VERSION = "sc-neurocore.scnir.v0.2"
SCNIR_LEGACY_SCHEMA_VERSION = "sc-neurocore.scnir.v0.1"
SCNIR_SUPPORTED_SCHEMA_VERSIONS = frozenset(
    {
        SCNIR_LEGACY_SCHEMA_VERSION,
        SCNIR_V02_SCHEMA_VERSION,
        SCNIR_V03_SCHEMA_VERSION,
        SCNIR_V04_SCHEMA_VERSION,
        SCNIR_PREVIOUS_SCHEMA_VERSION,
        SCNIR_V06_SCHEMA_VERSION,
        SCNIR_SCHEMA_VERSION,
    }
)

SCNIREncoding = Literal[
    "unipolar",
    "bipolar",
    "low_discrepancy",
    "deterministic_replay",
    "stochastic_lfsr",
    "hardware_source",
]
SCNIRRounding = Literal["nearest_even", "towards_zero", "stochastic", "floor", "ceil"]
SCNIROverflow = Literal["saturate", "wrap", "error"]
SCNIRSourceKind = Literal["lfsr", "sobol", "halton", "replay", "hardware"]
SCNIRSignalKind = Literal["spike", "analogue_state", "weight"]
SCNIRTransformKind = Literal["threshold"]
SCNIRTransformPosition = Literal["source", "destination"]
SCNIRComparison = Literal["greater_than"]
SCNIRHierarchyPortDirection = Literal["input", "output", "inout"]
SCNIRCorrelationPolicy = Literal[
    "independent",
    "must_share_source",
    "must_decorrelate",
    "max_correlation",
    "seed_isolation_domain",
]
SCNIRDelaySteps = int | Sequence[int]

_ENCODINGS = frozenset(
    {
        "unipolar",
        "bipolar",
        "low_discrepancy",
        "deterministic_replay",
        "stochastic_lfsr",
        "hardware_source",
    }
)
_ROUNDING_MODES = frozenset({"nearest_even", "towards_zero", "stochastic", "floor", "ceil"})
_OVERFLOW_MODES = frozenset({"saturate", "wrap", "error"})
_SOURCE_KINDS = frozenset({"lfsr", "sobol", "halton", "replay", "hardware"})
_SIGNAL_KINDS = frozenset({"spike", "analogue_state", "weight"})
_TRANSFORM_KINDS = frozenset({"threshold"})
_TRANSFORM_POSITIONS = frozenset({"source", "destination"})
_COMPARISONS = frozenset({"greater_than"})
_HIERARCHY_DIRECTIONS = frozenset({"input", "output", "inout"})
_CORRELATION_POLICIES = frozenset(
    {
        "independent",
        "must_share_source",
        "must_decorrelate",
        "max_correlation",
        "seed_isolation_domain",
    }
)
_STREAM_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_HDL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_MAX_SEED = (1 << 64) - 1


class SCNIRValidationError(ValueError):
    """Raised when an SC-NIR payload violates the fail-closed contract."""


@dataclass(frozen=True, slots=True)
class SCNIRPrecision:
    """Fixed-point interpretation attached to one stochastic stream."""

    signed: bool
    total_bits: int
    fractional_bits: int
    accumulator_bits: int
    rounding: SCNIRRounding
    overflow: SCNIROverflow


@dataclass(frozen=True, slots=True)
class SCNIRSource:
    """Random or deterministic source metadata for a stochastic stream."""

    kind: SCNIRSourceKind
    seed: int | None = None
    lfsr_polynomial: str | None = None
    tap_mask: int | None = None
    sobol_dimension: int | None = None
    halton_base: int | None = None
    replay_uri: str | None = None
    hardware_id: str | None = None


@dataclass(frozen=True, slots=True)
class SCNIRCorrelationConstraint:
    """Correlation rule between two stochastic streams."""

    peer_stream_id: str
    policy: SCNIRCorrelationPolicy
    max_abs_correlation: float | None = None
    seed_domain: str | None = None


@dataclass(frozen=True, slots=True)
class SCNIRStreamTransform:
    """Deterministic transform applied before a logical stochastic stream."""

    kind: SCNIRTransformKind
    position: SCNIRTransformPosition
    comparison: SCNIRComparison
    values: Sequence[float]


@dataclass(frozen=True, slots=True)
class SCNIRStream:
    """SC metadata for one logical stochastic bitstream."""

    stream_id: str
    layer: str
    bitstream_length: int
    encoding: SCNIREncoding
    precision: SCNIRPrecision
    source: SCNIRSource
    signal_kind: SCNIRSignalKind = "spike"
    delay_steps: SCNIRDelaySteps = 0
    transforms: Sequence[SCNIRStreamTransform] = field(default_factory=tuple)
    correlation_constraints: Sequence[SCNIRCorrelationConstraint] = field(default_factory=tuple)
    online_learning: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SCNIRHierarchyPort:
    """One typed port on a hierarchical SC-NIR hardware instance."""

    port_name: str
    direction: SCNIRHierarchyPortDirection
    stream_id: str
    signal_kind: SCNIRSignalKind
    bit_width: int


@dataclass(frozen=True, slots=True)
class SCNIRHierarchyInstance:
    """One hierarchy instance boundary for future nested hardware handoff."""

    instance_id: str
    module_name: str
    ports: Sequence[SCNIRHierarchyPort]


@dataclass(frozen=True, slots=True)
class SCNIRDocument:
    """Top-level SC-NIR metadata document."""

    producer: str
    streams: Sequence[SCNIRStream]
    hierarchy: Sequence[SCNIRHierarchyInstance] = field(default_factory=tuple)
    schema_version: str = SCNIR_SCHEMA_VERSION


def validate_scnir_dict(payload: Mapping[str, Any]) -> None:
    """Validate a decoded SC-NIR payload or raise ``SCNIRValidationError``."""

    _expect_keys(payload, {"schema_version", "producer", "streams", "hierarchy"}, "document")
    if payload["schema_version"] != SCNIR_SCHEMA_VERSION:
        raise SCNIRValidationError(
            f"schema_version must be {SCNIR_SCHEMA_VERSION!r}, got {payload['schema_version']!r}"
        )
    _expect_non_empty_string(payload["producer"], "producer")
    streams = _expect_sequence(payload["streams"], "streams")
    if not streams:
        raise SCNIRValidationError("streams must contain at least one stream")

    stream_ids: set[str] = set()
    stream_signal_kinds: dict[str, SCNIRSignalKind] = {}
    stream_payloads: list[Mapping[str, Any]] = []
    for index, item in enumerate(streams):
        stream = _expect_mapping(item, f"streams[{index}]")
        _validate_stream(stream, f"streams[{index}]")
        stream_id = cast(str, stream["stream_id"])
        if stream_id in stream_ids:
            raise SCNIRValidationError(f"duplicate stream_id {stream_id!r}")
        stream_ids.add(stream_id)
        stream_signal_kinds[stream_id] = cast(SCNIRSignalKind, stream["signal_kind"])
        stream_payloads.append(stream)

    for index, stream in enumerate(stream_payloads):
        constraints = _expect_sequence(
            stream.get("correlation_constraints", ()), f"streams[{index}].correlation_constraints"
        )
        for c_index, item in enumerate(constraints):
            constraint = _expect_mapping(
                item, f"streams[{index}].correlation_constraints[{c_index}]"
            )
            peer = cast(str, constraint["peer_stream_id"])
            if peer not in stream_ids:
                raise SCNIRValidationError(
                    f"streams[{index}].correlation_constraints[{c_index}].peer_stream_id "
                    f"{peer!r} does not reference an existing stream"
                )

    _validate_hierarchy(
        _expect_sequence(payload["hierarchy"], "hierarchy"),
        stream_signal_kinds,
    )


def scnir_from_dict(payload: Mapping[str, Any]) -> SCNIRDocument:
    """Build a typed SC-NIR document from a decoded mapping."""

    validate_scnir_dict(payload)
    streams_payload = _expect_sequence(payload["streams"], "streams")
    streams = tuple(
        _stream_from_dict(_expect_mapping(item, f"streams[{index}]"))
        for index, item in enumerate(streams_payload)
    )
    return SCNIRDocument(
        schema_version=cast(str, payload["schema_version"]),
        producer=cast(str, payload["producer"]),
        streams=streams,
        hierarchy=tuple(
            _hierarchy_instance_from_dict(_expect_mapping(item, f"hierarchy[{index}]"))
            for index, item in enumerate(_expect_sequence(payload["hierarchy"], "hierarchy"))
        ),
    )


def scnir_to_dict(document: SCNIRDocument) -> dict[str, Any]:
    """Convert a typed SC-NIR document to deterministic JSON-ready data."""

    payload: dict[str, Any] = {
        "schema_version": document.schema_version,
        "producer": document.producer,
        "streams": [_stream_to_dict(stream) for stream in document.streams],
        "hierarchy": [_hierarchy_instance_to_dict(instance) for instance in document.hierarchy],
    }
    validate_scnir_dict(payload)
    return payload


def load_scnir(path: str | Path) -> SCNIRDocument:
    """Load and validate an SC-NIR JSON document."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return scnir_from_dict(_expect_mapping(raw, "document"))


def write_scnir(path: str | Path, document: SCNIRDocument) -> None:
    """Write an SC-NIR JSON document after validating it."""

    payload = scnir_to_dict(document)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def upgrade_scnir_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade supported SC-NIR payloads to the current canonical schema.

    Version ``v0.1`` did not encode recurrent connection delay, and versions
    before ``v0.3`` did not distinguish spiking, analogue-state, and weight
    streams.  Version ``v0.4`` added explicit stream transform metadata for
    threshold comparators.  Version ``v0.5`` permits ``delay_steps`` to be
    either a scalar integer or a per-source-column integer vector.  Version
    ``v0.6`` adds top-level hierarchy instance and port metadata.  Version
    ``v0.7`` adds optional validated per-weight-stream online-learning
    annotations.  Legacy upgrades insert the missing fields before validating
    through the typed schema.  Current documents are canonicalised through the
    same deterministic writer.
    """

    version = payload.get("schema_version")
    if not isinstance(version, str) or version not in SCNIR_SUPPORTED_SCHEMA_VERSIONS:
        raise SCNIRValidationError(f"unsupported SC-NIR schema_version {version!r}")
    if version in {
        SCNIR_LEGACY_SCHEMA_VERSION,
        SCNIR_V02_SCHEMA_VERSION,
        SCNIR_V03_SCHEMA_VERSION,
        SCNIR_V04_SCHEMA_VERSION,
        SCNIR_PREVIOUS_SCHEMA_VERSION,
        SCNIR_V06_SCHEMA_VERSION,
    }:
        upgraded: dict[str, Any] = dict(payload)
        upgraded["schema_version"] = SCNIR_SCHEMA_VERSION
        streams = _expect_sequence(upgraded.get("streams"), "streams")
        upgraded_streams: list[dict[str, Any]] = []
        for index, stream in enumerate(streams):
            stream_payload = dict(_expect_mapping(stream, f"streams[{index}]"))
            if "delay_steps" not in stream_payload:
                stream_payload["delay_steps"] = 0
            if "signal_kind" not in stream_payload:
                stream_payload["signal_kind"] = _infer_legacy_signal_kind(
                    str(stream_payload.get("stream_id", ""))
                )
            if "transforms" not in stream_payload:
                stream_payload["transforms"] = []
            if "online_learning" not in stream_payload:
                stream_payload["online_learning"] = None
            upgraded_streams.append(stream_payload)
        upgraded["streams"] = upgraded_streams
        if "hierarchy" not in upgraded:
            upgraded["hierarchy"] = []
        return scnir_to_dict(scnir_from_dict(upgraded))
    return scnir_to_dict(scnir_from_dict(payload))


def _validate_stream(stream: Mapping[str, Any], path: str) -> None:
    _expect_keys(
        stream,
        {
            "stream_id",
            "layer",
            "bitstream_length",
            "encoding",
            "signal_kind",
            "precision",
            "source",
            "delay_steps",
            "transforms",
            "correlation_constraints",
            "online_learning",
        },
        path,
    )
    _expect_stream_id(stream["stream_id"], f"{path}.stream_id")
    _expect_non_empty_string(stream["layer"], f"{path}.layer")
    _expect_positive_int(stream["bitstream_length"], f"{path}.bitstream_length")
    _expect_enum(stream["encoding"], _ENCODINGS, f"{path}.encoding")
    _expect_enum(stream["signal_kind"], _SIGNAL_KINDS, f"{path}.signal_kind")
    _validate_precision(_expect_mapping(stream["precision"], f"{path}.precision"), path)
    _validate_source(_expect_mapping(stream["source"], f"{path}.source"), path)
    _expect_delay_steps(stream["delay_steps"], f"{path}.delay_steps")
    transforms = _expect_sequence(stream["transforms"], f"{path}.transforms")
    for index, item in enumerate(transforms):
        _validate_transform(_expect_mapping(item, f"{path}.transforms[{index}]"), path)
    constraints = _expect_sequence(
        stream["correlation_constraints"], f"{path}.correlation_constraints"
    )
    for index, item in enumerate(constraints):
        _validate_correlation(
            _expect_mapping(item, f"{path}.correlation_constraints[{index}]"),
            f"{path}.correlation_constraints[{index}]",
        )
    online_learning = stream["online_learning"]
    if online_learning is not None:
        if stream["signal_kind"] != "weight":
            raise SCNIRValidationError(f"{path}.online_learning is only valid on weight streams")
        _validate_online_learning_annotation(
            _expect_mapping(online_learning, f"{path}.online_learning"),
            f"{path}.online_learning",
        )


def _validate_online_learning_annotation(annotation: Mapping[str, Any], path: str) -> None:
    _expect_keys(
        annotation,
        {
            "schema_version",
            "rule_id",
            "rule_family",
            "state_fields",
            "per_synapse_state_bits",
            "weight_bits",
            "trace_bits",
            "reward_bits",
            "learning_shift",
            "trace_decay_shift",
            "saturation_policy",
            "hidden_history_fields",
            "sequence_length_independent",
        },
        path,
    )
    if annotation["schema_version"] != "sc-neurocore.online-o1.annotation.v1":
        raise SCNIRValidationError(f"{path}.schema_version is unsupported")
    _expect_non_empty_string(annotation["rule_id"], f"{path}.rule_id")
    if annotation["rule_family"] != "reward_modulated_stdp":
        raise SCNIRValidationError(f"{path}.rule_family is unsupported")
    state_fields = tuple(_expect_sequence(annotation["state_fields"], f"{path}.state_fields"))
    if state_fields != ("weight", "pre_trace", "post_trace", "eligibility"):
        raise SCNIRValidationError(f"{path}.state_fields must match the Online O(1) contract")
    weight_bits = _expect_positive_int(annotation["weight_bits"], f"{path}.weight_bits")
    trace_bits = _expect_positive_int(annotation["trace_bits"], f"{path}.trace_bits")
    if trace_bits < 2:
        raise SCNIRValidationError(f"{path}.trace_bits must be >= 2")
    _expect_positive_int(annotation["reward_bits"], f"{path}.reward_bits")
    _expect_non_negative_int(annotation["learning_shift"], f"{path}.learning_shift")
    _expect_non_negative_int(annotation["trace_decay_shift"], f"{path}.trace_decay_shift")
    expected_state_bits = weight_bits + 3 * trace_bits
    if annotation["per_synapse_state_bits"] != expected_state_bits:
        raise SCNIRValidationError(
            f"{path}.per_synapse_state_bits must equal weight_bits + 3 * trace_bits"
        )
    if annotation["saturation_policy"] != "signed_eligibility_unsigned_weight":
        raise SCNIRValidationError(f"{path}.saturation_policy is unsupported")
    hidden_history_fields = _expect_sequence(
        annotation["hidden_history_fields"], f"{path}.hidden_history_fields"
    )
    if hidden_history_fields:
        raise SCNIRValidationError(f"{path}.hidden_history_fields must be empty")
    if annotation["sequence_length_independent"] is not True:
        raise SCNIRValidationError(f"{path}.sequence_length_independent must be true")


def _validate_transform(transform: Mapping[str, Any], parent_path: str) -> None:
    path = f"{parent_path}.transforms"
    _expect_keys(transform, {"kind", "position", "comparison", "values"}, path)
    _expect_enum(transform["kind"], _TRANSFORM_KINDS, f"{path}.kind")
    _expect_enum(transform["position"], _TRANSFORM_POSITIONS, f"{path}.position")
    _expect_enum(transform["comparison"], _COMPARISONS, f"{path}.comparison")
    values = _expect_sequence(transform["values"], f"{path}.values")
    if not values:
        raise SCNIRValidationError(f"{path}.values must contain at least one threshold")
    for index, value in enumerate(values):
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise SCNIRValidationError(f"{path}.values[{index}] must be numeric")
        if not math.isfinite(float(value)):
            raise SCNIRValidationError(f"{path}.values[{index}] must be finite")


def _validate_precision(precision: Mapping[str, Any], parent_path: str) -> None:
    path = f"{parent_path}.precision"
    _expect_keys(
        precision,
        {"signed", "total_bits", "fractional_bits", "accumulator_bits", "rounding", "overflow"},
        path,
    )
    if not isinstance(precision["signed"], bool):
        raise SCNIRValidationError(f"{path}.signed must be a boolean")
    total_bits = _expect_positive_int(precision["total_bits"], f"{path}.total_bits")
    fractional_bits = _expect_non_negative_int(
        precision["fractional_bits"], f"{path}.fractional_bits"
    )
    accumulator_bits = _expect_positive_int(
        precision["accumulator_bits"], f"{path}.accumulator_bits"
    )
    if fractional_bits >= total_bits:
        raise SCNIRValidationError(f"{path}.fractional_bits must be smaller than total_bits")
    if accumulator_bits < total_bits:
        raise SCNIRValidationError(f"{path}.accumulator_bits must be >= total_bits")
    _expect_enum(precision["rounding"], _ROUNDING_MODES, f"{path}.rounding")
    _expect_enum(precision["overflow"], _OVERFLOW_MODES, f"{path}.overflow")


def _validate_source(source: Mapping[str, Any], parent_path: str) -> None:
    path = f"{parent_path}.source"
    _expect_keys(
        source,
        {
            "kind",
            "seed",
            "lfsr_polynomial",
            "tap_mask",
            "sobol_dimension",
            "halton_base",
            "replay_uri",
            "hardware_id",
        },
        path,
    )
    kind = cast(SCNIRSourceKind, _expect_enum(source["kind"], _SOURCE_KINDS, f"{path}.kind"))
    seed = source["seed"]
    if seed is not None:
        seed_int = _expect_non_negative_int(seed, f"{path}.seed")
        if seed_int > _MAX_SEED:
            raise SCNIRValidationError(f"{path}.seed must fit in uint64")

    if kind == "lfsr":
        _expect_non_empty_string(source["lfsr_polynomial"], f"{path}.lfsr_polynomial")
        _expect_positive_int(source["tap_mask"], f"{path}.tap_mask")
    elif kind == "sobol":
        _expect_positive_int(source["sobol_dimension"], f"{path}.sobol_dimension")
    elif kind == "halton":
        base = _expect_positive_int(source["halton_base"], f"{path}.halton_base")
        if not _is_prime(base):
            raise SCNIRValidationError(f"{path}.halton_base must be prime")
    elif kind == "replay":
        _expect_non_empty_string(source["replay_uri"], f"{path}.replay_uri")
    elif kind == "hardware":
        _expect_non_empty_string(source["hardware_id"], f"{path}.hardware_id")


def _validate_correlation(constraint: Mapping[str, Any], path: str) -> None:
    _expect_keys(
        constraint,
        {"peer_stream_id", "policy", "max_abs_correlation", "seed_domain"},
        path,
    )
    _expect_stream_id(constraint["peer_stream_id"], f"{path}.peer_stream_id")
    policy = cast(
        SCNIRCorrelationPolicy,
        _expect_enum(constraint["policy"], _CORRELATION_POLICIES, f"{path}.policy"),
    )
    max_abs_correlation = constraint["max_abs_correlation"]
    if max_abs_correlation is not None:
        if not isinstance(max_abs_correlation, int | float) or isinstance(
            max_abs_correlation, bool
        ):
            raise SCNIRValidationError(f"{path}.max_abs_correlation must be numeric")
        if not 0.0 <= float(max_abs_correlation) <= 1.0:
            raise SCNIRValidationError(f"{path}.max_abs_correlation must be in [0, 1]")
    if policy == "max_correlation" and max_abs_correlation is None:
        raise SCNIRValidationError(f"{path}.max_abs_correlation is required")
    seed_domain = constraint["seed_domain"]
    if seed_domain is not None:
        _expect_non_empty_string(seed_domain, f"{path}.seed_domain")


def _validate_hierarchy(
    hierarchy: Sequence[Any],
    stream_signal_kinds: Mapping[str, SCNIRSignalKind],
) -> None:
    instance_ids: set[str] = set()
    for index, item in enumerate(hierarchy):
        path = f"hierarchy[{index}]"
        instance = _expect_mapping(item, path)
        _expect_keys(instance, {"instance_id", "module_name", "ports"}, path)
        instance_id = _expect_stream_id(instance["instance_id"], f"{path}.instance_id")
        if instance_id in instance_ids:
            raise SCNIRValidationError(f"duplicate hierarchy instance_id {instance_id!r}")
        instance_ids.add(instance_id)
        _expect_hdl_identifier(instance["module_name"], f"{path}.module_name")

        ports = _expect_sequence(instance["ports"], f"{path}.ports")
        if not ports:
            raise SCNIRValidationError(f"{path}.ports must contain at least one port")
        port_names: set[str] = set()
        for port_index, port_item in enumerate(ports):
            port_path = f"{path}.ports[{port_index}]"
            port = _expect_mapping(port_item, port_path)
            _expect_keys(
                port,
                {"port_name", "direction", "stream_id", "signal_kind", "bit_width"},
                port_path,
            )
            port_name = _expect_hdl_identifier(port["port_name"], f"{port_path}.port_name")
            if port_name in port_names:
                raise SCNIRValidationError(
                    f"{path}.ports contains duplicate port_name {port_name!r}"
                )
            port_names.add(port_name)
            _expect_enum(port["direction"], _HIERARCHY_DIRECTIONS, f"{port_path}.direction")
            stream_id = _expect_stream_id(port["stream_id"], f"{port_path}.stream_id")
            if stream_id not in stream_signal_kinds:
                raise SCNIRValidationError(
                    f"{port_path}.stream_id {stream_id!r} does not reference an existing stream"
                )
            signal_kind = cast(
                SCNIRSignalKind,
                _expect_enum(port["signal_kind"], _SIGNAL_KINDS, f"{port_path}.signal_kind"),
            )
            if signal_kind != stream_signal_kinds[stream_id]:
                raise SCNIRValidationError(
                    f"{port_path}.signal_kind {signal_kind!r} does not match stream "
                    f"{stream_id!r} signal_kind {stream_signal_kinds[stream_id]!r}"
                )
            _expect_positive_int(port["bit_width"], f"{port_path}.bit_width")


def _stream_from_dict(stream: Mapping[str, Any]) -> SCNIRStream:
    precision = _expect_mapping(stream["precision"], "precision")
    source = _expect_mapping(stream["source"], "source")
    transforms = _expect_sequence(stream["transforms"], "transforms")
    constraints = _expect_sequence(stream["correlation_constraints"], "correlation_constraints")
    return SCNIRStream(
        stream_id=cast(str, stream["stream_id"]),
        layer=cast(str, stream["layer"]),
        bitstream_length=cast(int, stream["bitstream_length"]),
        encoding=cast(SCNIREncoding, stream["encoding"]),
        signal_kind=cast(SCNIRSignalKind, stream["signal_kind"]),
        precision=SCNIRPrecision(
            signed=cast(bool, precision["signed"]),
            total_bits=cast(int, precision["total_bits"]),
            fractional_bits=cast(int, precision["fractional_bits"]),
            accumulator_bits=cast(int, precision["accumulator_bits"]),
            rounding=cast(SCNIRRounding, precision["rounding"]),
            overflow=cast(SCNIROverflow, precision["overflow"]),
        ),
        source=SCNIRSource(
            kind=cast(SCNIRSourceKind, source["kind"]),
            seed=cast(int | None, source["seed"]),
            lfsr_polynomial=cast(str | None, source["lfsr_polynomial"]),
            tap_mask=cast(int | None, source["tap_mask"]),
            sobol_dimension=cast(int | None, source["sobol_dimension"]),
            halton_base=cast(int | None, source["halton_base"]),
            replay_uri=cast(str | None, source["replay_uri"]),
            hardware_id=cast(str | None, source["hardware_id"]),
        ),
        delay_steps=_delay_steps_from_value(stream["delay_steps"], "stream.delay_steps"),
        transforms=tuple(
            _transform_from_dict(_expect_mapping(item, "transform")) for item in transforms
        ),
        correlation_constraints=tuple(
            _correlation_from_dict(_expect_mapping(item, "correlation_constraint"))
            for item in constraints
        ),
        online_learning=(
            dict(_expect_mapping(stream["online_learning"], "stream.online_learning"))
            if stream["online_learning"] is not None
            else None
        ),
    )


def _transform_from_dict(transform: Mapping[str, Any]) -> SCNIRStreamTransform:
    values = _expect_sequence(transform["values"], "transform.values")
    return SCNIRStreamTransform(
        kind=cast(SCNIRTransformKind, transform["kind"]),
        position=cast(SCNIRTransformPosition, transform["position"]),
        comparison=cast(SCNIRComparison, transform["comparison"]),
        values=tuple(float(value) for value in values),
    )


def _correlation_from_dict(constraint: Mapping[str, Any]) -> SCNIRCorrelationConstraint:
    return SCNIRCorrelationConstraint(
        peer_stream_id=cast(str, constraint["peer_stream_id"]),
        policy=cast(SCNIRCorrelationPolicy, constraint["policy"]),
        max_abs_correlation=cast(float | None, constraint["max_abs_correlation"]),
        seed_domain=cast(str | None, constraint["seed_domain"]),
    )


def _hierarchy_instance_from_dict(instance: Mapping[str, Any]) -> SCNIRHierarchyInstance:
    ports = _expect_sequence(instance["ports"], "hierarchy.ports")
    return SCNIRHierarchyInstance(
        instance_id=cast(str, instance["instance_id"]),
        module_name=cast(str, instance["module_name"]),
        ports=tuple(
            _hierarchy_port_from_dict(_expect_mapping(port, "hierarchy.port")) for port in ports
        ),
    )


def _hierarchy_port_from_dict(port: Mapping[str, Any]) -> SCNIRHierarchyPort:
    return SCNIRHierarchyPort(
        port_name=cast(str, port["port_name"]),
        direction=cast(SCNIRHierarchyPortDirection, port["direction"]),
        stream_id=cast(str, port["stream_id"]),
        signal_kind=cast(SCNIRSignalKind, port["signal_kind"]),
        bit_width=cast(int, port["bit_width"]),
    )


def _stream_to_dict(stream: SCNIRStream) -> dict[str, Any]:
    return {
        "stream_id": stream.stream_id,
        "layer": stream.layer,
        "bitstream_length": stream.bitstream_length,
        "encoding": stream.encoding,
        "signal_kind": stream.signal_kind,
        "delay_steps": _delay_steps_to_json(stream.delay_steps),
        "precision": {
            "signed": stream.precision.signed,
            "total_bits": stream.precision.total_bits,
            "fractional_bits": stream.precision.fractional_bits,
            "accumulator_bits": stream.precision.accumulator_bits,
            "rounding": stream.precision.rounding,
            "overflow": stream.precision.overflow,
        },
        "source": {
            "kind": stream.source.kind,
            "seed": stream.source.seed,
            "lfsr_polynomial": stream.source.lfsr_polynomial,
            "tap_mask": stream.source.tap_mask,
            "sobol_dimension": stream.source.sobol_dimension,
            "halton_base": stream.source.halton_base,
            "replay_uri": stream.source.replay_uri,
            "hardware_id": stream.source.hardware_id,
        },
        "transforms": [
            {
                "kind": transform.kind,
                "position": transform.position,
                "comparison": transform.comparison,
                "values": [float(value) for value in transform.values],
            }
            for transform in stream.transforms
        ],
        "correlation_constraints": [
            {
                "peer_stream_id": constraint.peer_stream_id,
                "policy": constraint.policy,
                "max_abs_correlation": constraint.max_abs_correlation,
                "seed_domain": constraint.seed_domain,
            }
            for constraint in stream.correlation_constraints
        ],
        "online_learning": dict(stream.online_learning)
        if stream.online_learning is not None
        else None,
    }


def _hierarchy_instance_to_dict(instance: SCNIRHierarchyInstance) -> dict[str, Any]:
    return {
        "instance_id": instance.instance_id,
        "module_name": instance.module_name,
        "ports": [_hierarchy_port_to_dict(port) for port in instance.ports],
    }


def _hierarchy_port_to_dict(port: SCNIRHierarchyPort) -> dict[str, Any]:
    return {
        "port_name": port.port_name,
        "direction": port.direction,
        "stream_id": port.stream_id,
        "signal_kind": port.signal_kind,
        "bit_width": port.bit_width,
    }


def _infer_legacy_signal_kind(stream_id: str) -> SCNIRSignalKind:
    if stream_id.startswith("conn.") or stream_id.endswith((".weight", "_weight", "-weight")):
        return "weight"
    if stream_id.endswith(".state") or stream_id.endswith(".value"):
        return "analogue_state"
    return "spike"


def _expect_keys(payload: Mapping[str, Any], allowed: set[str], path: str) -> None:
    actual = set(payload)
    missing = allowed - actual
    unknown = actual - allowed
    if missing:
        raise SCNIRValidationError(f"{path} missing required field(s): {sorted(missing)}")
    if unknown:
        raise SCNIRValidationError(f"{path} contains unknown field(s): {sorted(unknown)}")


def _expect_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SCNIRValidationError(f"{path} must be an object")
    return value


def _expect_sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise SCNIRValidationError(f"{path} must be an array")
    return value


def _expect_non_empty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SCNIRValidationError(f"{path} must be a non-empty string")
    return value


def _expect_stream_id(value: Any, path: str) -> str:
    stream_id = _expect_non_empty_string(value, path)
    if _STREAM_ID_RE.fullmatch(stream_id) is None:
        raise SCNIRValidationError(f"{path} has invalid identifier syntax")
    return stream_id


def _expect_hdl_identifier(value: Any, path: str) -> str:
    identifier = _expect_non_empty_string(value, path)
    if _HDL_IDENTIFIER_RE.fullmatch(identifier) is None:
        raise SCNIRValidationError(f"{path} has invalid HDL identifier syntax")
    return identifier


def _expect_positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SCNIRValidationError(f"{path} must be a positive integer")
    return value


def _expect_non_negative_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SCNIRValidationError(f"{path} must be a non-negative integer")
    return value


def _expect_delay_steps(value: Any, path: str) -> int | tuple[int, ...]:
    """Validate scalar or per-source-column stream delay metadata."""

    if isinstance(value, int) and not isinstance(value, bool):
        return _expect_non_negative_int(value, path)
    sequence = _expect_sequence(value, path)
    if not sequence:
        raise SCNIRValidationError(f"{path} vector must contain at least one value")
    steps: list[int] = []
    for index, item in enumerate(sequence):
        steps.append(_expect_non_negative_int(item, f"{path}[{index}]"))
    return tuple(steps)


def _delay_steps_from_value(value: Any, path: str) -> int | tuple[int, ...]:
    return _expect_delay_steps(value, path)


def _delay_steps_to_json(value: SCNIRDelaySteps) -> int | list[int]:
    steps = _expect_delay_steps(value, "delay_steps")
    if isinstance(steps, int):
        return steps
    return list(steps)


def _expect_enum(value: Any, allowed: frozenset[str], path: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise SCNIRValidationError(f"{path} must be one of {sorted(allowed)}")
    return value


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True
