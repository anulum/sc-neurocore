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
metadata, and correlation constraints.  Validation is intentionally fail-closed
so unrecognised or under-specified metadata cannot silently reach hardware
generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence, cast

SCNIR_SCHEMA_VERSION = "sc-neurocore.scnir.v0.1"
SCNIR_SUPPORTED_SCHEMA_VERSIONS = frozenset({SCNIR_SCHEMA_VERSION})

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
SCNIRCorrelationPolicy = Literal[
    "independent",
    "must_share_source",
    "must_decorrelate",
    "max_correlation",
    "seed_isolation_domain",
]

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
class SCNIRStream:
    """SC metadata for one logical stochastic bitstream."""

    stream_id: str
    layer: str
    bitstream_length: int
    encoding: SCNIREncoding
    precision: SCNIRPrecision
    source: SCNIRSource
    correlation_constraints: Sequence[SCNIRCorrelationConstraint] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class SCNIRDocument:
    """Top-level SC-NIR metadata document."""

    producer: str
    streams: Sequence[SCNIRStream]
    schema_version: str = SCNIR_SCHEMA_VERSION


def validate_scnir_dict(payload: Mapping[str, Any]) -> None:
    """Validate a decoded SC-NIR payload or raise ``SCNIRValidationError``."""

    _expect_keys(payload, {"schema_version", "producer", "streams"}, "document")
    if payload["schema_version"] != SCNIR_SCHEMA_VERSION:
        raise SCNIRValidationError(
            f"schema_version must be {SCNIR_SCHEMA_VERSION!r}, got {payload['schema_version']!r}"
        )
    _expect_non_empty_string(payload["producer"], "producer")
    streams = _expect_sequence(payload["streams"], "streams")
    if not streams:
        raise SCNIRValidationError("streams must contain at least one stream")

    stream_ids: set[str] = set()
    stream_payloads: list[Mapping[str, Any]] = []
    for index, item in enumerate(streams):
        stream = _expect_mapping(item, f"streams[{index}]")
        _validate_stream(stream, f"streams[{index}]")
        stream_id = cast(str, stream["stream_id"])
        if stream_id in stream_ids:
            raise SCNIRValidationError(f"duplicate stream_id {stream_id!r}")
        stream_ids.add(stream_id)
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
    )


def scnir_to_dict(document: SCNIRDocument) -> dict[str, Any]:
    """Convert a typed SC-NIR document to deterministic JSON-ready data."""

    payload: dict[str, Any] = {
        "schema_version": document.schema_version,
        "producer": document.producer,
        "streams": [_stream_to_dict(stream) for stream in document.streams],
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

    The current release only has one public schema version, so the migration is
    an identity migration through the typed validator. Keeping this as an
    explicit API gives future schema revisions a fail-closed migration surface
    instead of letting callers guess whether a document is silently accepted.
    """

    version = payload.get("schema_version")
    if not isinstance(version, str) or version not in SCNIR_SUPPORTED_SCHEMA_VERSIONS:
        raise SCNIRValidationError(f"unsupported SC-NIR schema_version {version!r}")
    return scnir_to_dict(scnir_from_dict(payload))


def _validate_stream(stream: Mapping[str, Any], path: str) -> None:
    _expect_keys(
        stream,
        {
            "stream_id",
            "layer",
            "bitstream_length",
            "encoding",
            "precision",
            "source",
            "correlation_constraints",
        },
        path,
    )
    _expect_stream_id(stream["stream_id"], f"{path}.stream_id")
    _expect_non_empty_string(stream["layer"], f"{path}.layer")
    _expect_positive_int(stream["bitstream_length"], f"{path}.bitstream_length")
    _expect_enum(stream["encoding"], _ENCODINGS, f"{path}.encoding")
    _validate_precision(_expect_mapping(stream["precision"], f"{path}.precision"), path)
    _validate_source(_expect_mapping(stream["source"], f"{path}.source"), path)
    constraints = _expect_sequence(
        stream["correlation_constraints"], f"{path}.correlation_constraints"
    )
    for index, item in enumerate(constraints):
        _validate_correlation(
            _expect_mapping(item, f"{path}.correlation_constraints[{index}]"),
            f"{path}.correlation_constraints[{index}]",
        )


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


def _stream_from_dict(stream: Mapping[str, Any]) -> SCNIRStream:
    precision = _expect_mapping(stream["precision"], "precision")
    source = _expect_mapping(stream["source"], "source")
    constraints = _expect_sequence(stream["correlation_constraints"], "correlation_constraints")
    return SCNIRStream(
        stream_id=cast(str, stream["stream_id"]),
        layer=cast(str, stream["layer"]),
        bitstream_length=cast(int, stream["bitstream_length"]),
        encoding=cast(SCNIREncoding, stream["encoding"]),
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
        correlation_constraints=tuple(
            _correlation_from_dict(_expect_mapping(item, "correlation_constraint"))
            for item in constraints
        ),
    )


def _correlation_from_dict(constraint: Mapping[str, Any]) -> SCNIRCorrelationConstraint:
    return SCNIRCorrelationConstraint(
        peer_stream_id=cast(str, constraint["peer_stream_id"]),
        policy=cast(SCNIRCorrelationPolicy, constraint["policy"]),
        max_abs_correlation=cast(float | None, constraint["max_abs_correlation"]),
        seed_domain=cast(str | None, constraint["seed_domain"]),
    )


def _stream_to_dict(stream: SCNIRStream) -> dict[str, Any]:
    return {
        "stream_id": stream.stream_id,
        "layer": stream.layer,
        "bitstream_length": stream.bitstream_length,
        "encoding": stream.encoding,
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
        "correlation_constraints": [
            {
                "peer_stream_id": constraint.peer_stream_id,
                "policy": constraint.policy,
                "max_abs_correlation": constraint.max_abs_correlation,
                "seed_domain": constraint.seed_domain,
            }
            for constraint in stream.correlation_constraints
        ],
    }


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


def _expect_positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SCNIRValidationError(f"{path} must be a positive integer")
    return value


def _expect_non_negative_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SCNIRValidationError(f"{path} must be a non-negative integer")
    return value


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
