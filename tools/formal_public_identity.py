# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hash-bound public formal evidence identity conversion

"""Validate descriptive execution labels without changing native proof facts."""

from __future__ import annotations

from collections.abc import Iterator
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import cast

from tools.formal_evidence_validation import (
    confined_file,
    digest,
    object_fields,
    read_object,
    require,
)

SCHEMA_VERSION = "sc-neurocore.formal-public-identity-conversion.v2"
PRIMARY_UNIT = "<FORMAL_PRIMARY_UNIT>"
AUDIT_UNIT = "<FORMAL_AUDIT_UNIT>"
_IDENTITY_NAME = re.compile(r"-(?:[0-9a-f]{8}|codex[^ /]*|claude[^ /]*)-20\d{6}T")
_CHANGES = {
    "receipt": {"measurement.unit", "custody.released_nonce", "measurement.hostname"},
    "nonvacuity": {
        "measurement.hostname",
        "measurement.unit",
        "producer",
        "receipt_sha256",
        "receipt_conversion.public_parent_receipt_sha256",
    },
}


def _hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _field(value: dict[str, object], path: str) -> object:
    result: object = value
    for key in path.split("."):
        fields = object_fields(result)
        require(key in fields, f"Converted public identity field {path} is absent.")
        result = fields[key]
    return result


def _omit(value: dict[str, object], path: str) -> None:
    keys = path.split(".")
    target = value
    for key in keys[:-1]:
        target = cast(dict[str, object], target[key])
    target.pop(keys[-1], None)


def _references(entry: dict[str, object]) -> Iterator[tuple[str, str]]:
    for key, value in entry.items():
        if key in {"receipt", "nonvacuity", "review", "nonvacuity_review"}:
            require(isinstance(value, str) and bool(value), "Invalid public evidence reference.")
            yield key, str(value)
        elif isinstance(value, dict):
            yield from _references(object_fields(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield from _references(object_fields(item))


def _converted_record(repository: Path, record: dict[str, object]) -> tuple[str, str]:
    path = confined_file(repository, record.get("path"))
    name = path.relative_to(repository).as_posix()
    public_sha = digest(path)
    require(
        public_sha == record.get("public_artifact_sha256")
        and _is_digest(record.get("original_artifact_sha256"))
        and public_sha != record.get("original_artifact_sha256"),
        "Public identity conversion artifact binding differs.",
    )
    kind = record.get("kind")
    require(
        isinstance(kind, str) and kind in _CHANGES,
        "Unknown public identity conversion record kind.",
    )
    payload = read_object(path)
    preserved = copy.deepcopy(payload)
    changes = record.get("changes")
    require(isinstance(changes, list) and bool(changes), "Missing public identity field changes.")
    seen: set[str] = set()
    for raw in cast(list[object], changes):
        change = object_fields(raw)
        field = change.get("field")
        require(
            isinstance(field, str) and field in _CHANGES[str(kind)] and field not in seen,
            "Public identity conversion changes an unsupported or repeated field.",
        )
        field = str(field)
        seen.add(field)
        require(
            _is_digest(change.get("original_value_sha256")), "Missing original identity digest."
        )
        if field == "custody.released_nonce":
            custody = object_fields(payload.get("custody"))
            require(
                change.get("operation") == "remove"
                and change.get("public_value_sha256") is None
                and "released_nonce" not in custody
                and _is_digest(custody.get("lease_release_sha256")),
                "Public lease identity must retain its complete private release-receipt binding.",
            )
        elif field == "measurement.hostname":
            profile = _field(payload, "measurement.host_profile")
            require(
                change.get("operation") == "rename"
                and change.get("public_field") == "measurement.host_profile"
                and "hostname" not in object_fields(payload.get("measurement"))
                and isinstance(profile, str)
                and bool(profile.strip())
                and change.get("public_value_sha256") == _hash(profile)
                and change.get("public_value_sha256") != change.get("original_value_sha256"),
                "Public host profile differs from its declared private identity conversion.",
            )
            _omit(preserved, "measurement.host_profile")
        else:
            require(
                change.get("operation") == "replace"
                and change.get("public_value_sha256") == _hash(_field(payload, field)),
                "Public identity field value differs from its conversion.",
            )
        _omit(preserved, field)
    require(
        "measurement.hostname" in seen,
        "Public converted evidence lacks its explicit host profile conversion.",
    )
    require(
        kind != "receipt" or "custody.released_nonce" in seen,
        "Public receipt conversion omits the private lease identity removal.",
    )
    require(
        "measurement.unit" in seen
        or (
            kind == "receipt"
            and object_fields(payload.get("measurement")).get("unit_accounting")
            == "garbage-collected"
            and "unit" not in object_fields(payload.get("measurement"))
        ),
        "Missing public unit conversion or invented historical unit identity.",
    )
    if "measurement.unit" in seen:
        require(
            _field(payload, "measurement.unit")
            == (AUDIT_UNIT if kind == "nonvacuity" else PRIMARY_UNIT),
            "Public primary and audit execution labels must remain distinct.",
        )
    if "producer" in seen:
        require(
            payload.get("producer") == "Native assertion-enable audit",
            "Public audit producer is not descriptive.",
        )
    require(
        _hash(preserved) == record.get("preserved_payload_sha256"),
        "Public identity conversion changed a preserved proof fact.",
    )
    return name, str(kind)


def validate_identity_conversion(repository: Path, entry: dict[str, object]) -> None:
    """Require explicit, hash-bound conversion of public execution identities.

    Parameters
    ----------
    repository : Path
        Current evidence checkout, including the independently reviewed manifest.
    entry : dict of str to object
        Indexed primary, audit, predecessor and review references.

    Raises
    ------
    ValueError
        If a coordination identity leaks, a conversion is absent or stale,
        a non-identity fact changes, or reviews omit the manifest binding.
    OSError
        If a referenced real evidence file is unavailable.

    Notes
    -----
    The existing native invocation, time-order, model, source, tool, execution
    accounting and independent-review gates still run after this check.
    Original identity values are retained privately; only their hashes appear
    in the public conversion manifest.
    """
    references = list(_references(entry))
    artifacts = [(kind, name) for kind, name in references if kind in _CHANGES]
    needs_conversion = False
    for _kind, name in artifacts:
        payload = read_object(confined_file(repository, name))
        measurement = object_fields(payload.get("measurement"))
        require(
            "hostname" not in measurement,
            "Public evidence contains a private hostname field.",
        )
        unit = measurement.get("unit")
        require(
            not isinstance(unit, str) or _IDENTITY_NAME.search(unit) is None,
            "Public evidence contains a private coordination unit identity.",
        )
        require(
            "released_nonce" not in object_fields(payload.get("custody", {})),
            "Public evidence contains a private lease identity.",
        )
        needs_conversion |= unit in (PRIMARY_UNIT, AUDIT_UNIT)
    binding = entry.get("identity_conversion")
    if binding is None:
        require(not needs_conversion, "Public execution label lacks its identity conversion.")
        return
    bound = object_fields(binding)
    path = confined_file(repository, bound.get("path"))
    manifest_sha = digest(path)
    require(manifest_sha == bound.get("sha256"), "Public identity manifest digest differs.")
    manifest = read_object(path)
    require(manifest.get("schema_version") == SCHEMA_VERSION, "Unknown public identity schema.")
    raw_records = manifest.get("records")
    require(isinstance(raw_records, list) and bool(raw_records), "Missing public identity records.")
    records: dict[str, str] = {}
    for raw in cast(list[object], raw_records):
        name, kind = _converted_record(repository, object_fields(raw))
        require(name not in records, "Repeated public identity artifact.")
        records[name] = kind
    require(
        all(records.get(name) == kind for kind, name in artifacts),
        "Public identity manifest omits an indexed evidence artifact.",
    )
    for kind, name in references:
        if kind in {"review", "nonvacuity_review"}:
            review = read_object(confined_file(repository, name))
            require(
                review.get("identity_conversion_sha256") == manifest_sha,
                "Independent review does not bind the public identity conversion.",
            )
