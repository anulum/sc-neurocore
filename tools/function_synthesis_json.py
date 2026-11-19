# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lossless bounded native netlist serialization

"""Keep original native JSON while admitting its lossless compact serialization."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import BinaryIO

MAX_NETLIST_BYTES = 64 * 1024 * 1024
_RAW = "mapped-native.json"
_RECORD = "mapped-json-normalization.json"
_TEMPORARY = "mapped-compact.json"
_SCHEMA = "sc-neurocore.native-json-whitespace.v1"

_SOURCE_TABLE_SCHEMA = "sc-neurocore.native-json-source-table.v1"
_SOURCE_TABLE_FIELDS = {"schema_version", "native", "source_locations"}
_JSON_STRING = re.compile(rb'"(?:[^"\\]++|\\.)*+"')
_SOURCE_KEY = re.compile(rb'"(?:s|\\u0073)(?:r|\\u0072)(?:c|\\u0063)"')
_JSON_WHITESPACE = b" \t\r\n"
_JSON_PUNCTUATION = b"{}[],:"
_SEPARATED_TOKENS = re.compile(rb"[^{}\[\],: \t\r\n][ \t\r\n]+(?=[^{}\[\],: \t\r\n])")
_CHUNK_BYTES = 65536


class _ParserBudgetExceeded(ValueError):
    """Mark a valid-to-this-point serialization that has outgrown the parser budget."""


def _check_budget(size: int) -> None:
    """Keep both native representations under the existing JSON parser cap."""
    if size > MAX_NETLIST_BYTES:
        raise _ParserBudgetExceeded("Normalized native JSON exceeds the unchanged parser budget")


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Refuse ambiguous original fields instead of silently retaining their last value."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> object:
    """Refuse nonstandard non-finite tokens in original native evidence."""
    raise ValueError(f"Non-finite JSON number: {value}")


def _finite_float(value: str) -> float:
    """Keep finite JSON fractions without allowing exponent overflow to become infinity."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite JSON number: {value}")
    return number


def decode_native_json(data: str | bytes) -> object:
    """Decode native evidence with unique keys and finite standard JSON values.

    Parameters
    ----------
    data : str or bytes
        Original evidence bytes or text already bounded by the caller.

    Returns
    -------
    object
        Exact parsed JSON value, without duplicate-key or non-finite coercion.

    Raises
    ------
    ValueError
        For malformed, ambiguous, nonstandard or excessively nested input.

    """
    try:
        value: object = json.loads(
            data,
            object_pairs_hook=_unique_keys,
            parse_constant=_invalid_constant,
            parse_float=_finite_float,
        )
    except RecursionError as error:
        raise ValueError("Native JSON exceeds the decoder nesting budget") from error
    return value


def decode_netlist_json(data: bytes) -> object:
    """Read a native JSON netlist and restore the exact values of its source metadata.

    Parameters
    ----------
    data : bytes
        Complete native or normalised netlist bytes within the parser budget.

    Returns
    -------
    object
        Original native JSON with any source-table references restored.

    Raises
    ------
    ValueError
        If the native document or its lossless source table is invalid.

    """
    value = decode_native_json(data)
    if not isinstance(value, dict) or value.get("schema_version") != _SOURCE_TABLE_SCHEMA:
        return value
    original: object = value.get("native")
    locations = value.get("source_locations")
    if (
        value.keys() != _SOURCE_TABLE_FIELDS
        or not isinstance(original, dict)
        or not isinstance(locations, list)
        or not all(isinstance(item, str) for item in locations)
        or len(locations) != len(set(locations))
    ):
        raise ValueError("The normalised netlist requires an exact native source table")
    pending: list[object] = [original]
    used: set[int] = set()
    while pending:
        record = pending.pop()
        if isinstance(record, dict):
            location = record.get("src")
            if type(location) is int:
                if not 0 <= location < len(locations):
                    raise ValueError("A native source reference is outside its retained table")
                record["src"] = locations[location]
                used.add(location)
            pending.extend(item for item in record.values() if isinstance(item, (dict, list)))
        elif isinstance(record, list):
            pending.extend(item for item in record if isinstance(item, (dict, list)))
    if used != set(range(len(locations))):
        raise ValueError("The normalised netlist contains unused native source locations")
    return original


def _digest(path: Path) -> str:
    """Hash every original byte without loading a complete artifact."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _regular(path: Path) -> None:
    """Require a real retained artifact instead of following a link."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("Native JSON normalization requires regular unlinked artifacts")


def _record(directory: Path, schema: str) -> dict[str, object]:
    """Describe the exact original and normalized representations."""
    original, compact = directory / _RAW, directory / "mapped.json"
    for path in (original, compact):
        _regular(path)
    _check_budget(compact.stat().st_size)
    return {
        "schema_version": schema,
        "transformation": (
            "remove-json-whitespace-outside-strings"
            if schema == _SCHEMA
            else "remove-json-whitespace-and-pool-native-source-locations"
        ),
        "original_sha256": _digest(original),
        "original_bytes": original.stat().st_size,
        "compact_sha256": _digest(compact),
        "compact_bytes": compact.stat().st_size,
        "parser_budget_bytes": MAX_NETLIST_BYTES,
    }


def _without_whitespace(parts: Iterable[tuple[bool, bytes]]) -> Iterator[bytes]:
    """Keep literals intact and refuse whitespace that separates primitive tokens."""
    separated = False
    previous: int | None = None
    for is_string, token in parts:
        if is_string:
            separated = False
            previous = None
            yield token
            continue
        body = token.lstrip(_JSON_WHITESPACE)
        if not body:
            separated = True
            continue
        leading_gap = len(body) != len(token)
        if (
            (separated or leading_gap)
            and previous is not None
            and previous not in _JSON_PUNCTUATION
            and body[0] not in _JSON_PUNCTUATION
        ) or _SEPARATED_TOKENS.search(body):
            raise ValueError("Native JSON whitespace separates distinct value tokens")
        content = body.rstrip(_JSON_WHITESPACE)
        previous = content[-1]
        separated = len(content) != len(body)
        yield content.translate(None, _JSON_WHITESPACE)


def _json_strings(source: BinaryIO) -> Iterator[tuple[bool, bytes]]:
    """Read JSON string tokens and intervening bytes without mapping mutable files."""
    block = source.read(_CHUNK_BYTES)
    cursor = 0
    while block:
        opening = block.find(b'"', cursor)
        if opening < 0:
            if cursor < len(block):
                yield False, block[cursor:]
            block, cursor = source.read(_CHUNK_BYTES), 0
            continue
        if opening > cursor:
            yield False, block[cursor:opening]
        match = _JSON_STRING.match(block, opening)
        while match is None:
            partial = block[opening:]
            _check_budget(len(partial) + 1)
            more = source.read(
                min(max(_CHUNK_BYTES, len(partial)), MAX_NETLIST_BYTES - len(partial))
            )
            if not more:
                raise ValueError("Native JSON normalization found an unterminated string")
            block, opening = partial + more, 0
            match = _JSON_STRING.match(block)
        cursor = match.end()
        yield True, match[0]


def _source_references(original: BinaryIO, locations: list[str]) -> Iterator[tuple[bool, bytes]]:
    """Replace native source strings while keeping all other JSON tokens exact."""
    indices: dict[str, int] = {}
    after_key = after_colon = False
    table_bytes = 0
    for is_string, token in _json_strings(original):
        if not is_string:
            if after_key:
                punctuation = token.lstrip(_JSON_WHITESPACE)
                if not after_colon and punctuation.startswith(b":"):
                    after_colon = True
                    punctuation = punctuation[1:].lstrip(_JSON_WHITESPACE)
                if punctuation:
                    if after_colon and punctuation[0] in b"-0123456789":
                        raise ValueError(
                            "Native source locations must be strings before normalization"
                        )
                    after_key = after_colon = False
            yield False, token
            continue
        if after_colon:
            value = decode_native_json(token)
            if not isinstance(value, str):
                raise ValueError("A native source location could not be decoded as a string")
            if value not in indices:
                table_bytes += len(json.dumps(value).encode("ascii")) + 1
                _check_budget(table_bytes)
                indices[value] = len(locations)
                locations.append(value)
            yield False, str(indices[value]).encode("ascii")
        else:
            yield True, token
        after_key = token == b'"src"' or (
            b"\\u" in token and _SOURCE_KEY.fullmatch(token) is not None
        )
        after_colon = False


def _source_table_document(original: Path) -> bytes:
    """Build one bounded reversible document without decoding the oversized original at once."""
    locations: list[str] = []
    output = bytearray(b'{"schema_version":"' + _SOURCE_TABLE_SCHEMA.encode() + b'","native":')
    with original.open("rb") as stream:
        for chunk in _without_whitespace(_source_references(stream, locations)):
            _check_budget(len(output) + len(chunk))
            output.extend(chunk)
    table = json.dumps(locations, separators=(",", ":")).encode("ascii")
    _check_budget(len(output) + len(table) + len(b',"source_locations":}'))
    output.extend(b',"source_locations":')
    output.extend(table)
    output.extend(b"}")
    return bytes(output)


def _write_normalized(original: Path, target: BinaryIO) -> str:
    """Try the existing byte-preserving representation before interning source locations."""
    try:
        with original.open("rb") as source:
            for chunk in _without_whitespace(_json_strings(source)):
                _check_budget(target.tell() + len(chunk))
                target.write(chunk)
    except _ParserBudgetExceeded:
        document = _source_table_document(original)
        decode_netlist_json(document)
        target.seek(0)
        target.write(document)
        target.truncate()
        return _SOURCE_TABLE_SCHEMA
    target.seek(0)
    decode_native_json(target.read(MAX_NETLIST_BYTES + 1))
    return _SCHEMA


def normalize_native_netlist(directory: Path) -> None:
    """Retain original native JSON and write a bounded, reversible compact copy.

    Parameters
    ----------
    directory : pathlib.Path
        Fresh completed native acquisition containing a regular mapped.json.
        The physical synthesis command and native cell statistics stay unchanged.

    Raises
    ------
    OSError, ValueError
        For linked, malformed, changing or already-normalized inputs, or a
        compact representation exceeding the unchanged 64 MiB parser budget.
        Failed attempts retain their original acquisition and partial output.

    """
    if directory.is_symlink():
        raise ValueError("Native JSON normalization must not follow a directory link")
    source = directory / "mapped.json"
    _regular(source)
    if any(
        (directory / name).exists() or (directory / name).is_symlink()
        for name in (_RAW, _RECORD, _TEMPORARY)
    ):
        raise ValueError("Native JSON normalization must not overwrite retained artifacts")
    before = _digest(source)
    with (directory / _TEMPORARY).open("x+b") as stream:
        schema = _write_normalized(source, stream)
    if _digest(source) != before:
        raise ValueError("The original native JSON changed during normalization")
    source.rename(directory / _RAW)
    (directory / _TEMPORARY).rename(source)
    (directory / _RECORD).write_text(
        json.dumps(_record(directory, schema), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def native_netlist_artifacts(directory: Path) -> tuple[str, ...]:
    """Validate retained normalization identities without rewriting either representation.

    Parameters
    ----------
    directory : pathlib.Path
        Original acquisition, with optional complete lossless-normalization
        record and the byte-for-byte native original.

    Returns
    -------
    tuple of str
        Both additional artifact names when normalization was performed.
        Original acquisitions without normalization return an empty tuple.

    Raises
    ------
    OSError, ValueError
        For linked, incomplete, oversized or changed normalization evidence.

    """
    if directory.is_symlink():
        raise ValueError("Native JSON normalization must not follow a directory link")
    record = directory / _RECORD
    if not record.exists() and not record.is_symlink():
        if (directory / _RAW).exists() or (directory / _RAW).is_symlink():
            raise ValueError("Native JSON normalization evidence is incomplete")
        return ()
    _regular(record)
    if record.stat().st_size > 65536:
        raise ValueError("The native JSON normalization record is oversized")
    value = decode_native_json(record.read_bytes())
    schema = value.get("schema_version") if isinstance(value, dict) else None
    if schema not in (_SCHEMA, _SOURCE_TABLE_SCHEMA) or value != _record(directory, schema):
        raise ValueError("The original or compact native JSON differs from its retained record")
    if schema == _SOURCE_TABLE_SCHEMA:
        expected = hashlib.sha256(_source_table_document(directory / _RAW)).hexdigest()
        if expected != value["compact_sha256"]:
            raise ValueError("The native source table differs from its complete original")
    return (_RAW, _RECORD)
