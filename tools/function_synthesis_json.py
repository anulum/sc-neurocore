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
from pathlib import Path

MAX_NETLIST_BYTES = 64 * 1024 * 1024
_RAW = "mapped-native.json"
_RECORD = "mapped-json-normalization.json"
_TEMPORARY = "mapped-compact.json"
_SCHEMA = "sc-neurocore.native-json-whitespace.v1"


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


def _digest(path: Path) -> str:
    """Hash every original byte without loading a complete artifact."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _regular(path: Path) -> None:
    """Require a real retained artifact instead of following a link."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("Native JSON normalization requires regular unlinked artifacts")


def _record(directory: Path) -> dict[str, object]:
    """Describe the exact original and normalized representations."""
    original, compact = directory / _RAW, directory / "mapped.json"
    for path in (original, compact):
        _regular(path)
    if compact.stat().st_size > MAX_NETLIST_BYTES:
        raise ValueError("Normalized native JSON exceeds the unchanged parser budget")
    return {
        "schema_version": _SCHEMA,
        "transformation": "remove-json-whitespace-outside-strings",
        "original_sha256": _digest(original),
        "original_bytes": original.stat().st_size,
        "compact_sha256": _digest(compact),
        "compact_bytes": compact.stat().st_size,
        "parser_budget_bytes": MAX_NETLIST_BYTES,
    }


def _compact(original: Path, target: Path) -> None:
    """Stream every JSON token and string byte with a bounded output budget."""
    quoted = escaped = False
    separated = False
    previous: int | None = None
    written = 0
    with original.open("rb") as source, target.open("xb") as destination:
        while chunk := source.read(65536):
            output = bytearray()
            for byte in chunk:
                if not quoted and byte in (9, 10, 13, 32):
                    separated = True
                    continue
                if (
                    separated
                    and previous is not None
                    and previous not in b'{}[],:"'
                    and byte not in b'{}[],:"'
                ):
                    raise ValueError("Native JSON whitespace separates distinct value tokens")
                separated = False
                if quoted:
                    output.append(byte)
                    if escaped:
                        escaped = False
                    elif byte == 92:
                        escaped = True
                    elif byte == 34:
                        quoted = False
                elif byte == 34:
                    quoted = True
                    output.append(byte)
                else:
                    output.append(byte)
                previous = byte
            written += len(output)
            if written > MAX_NETLIST_BYTES:
                raise ValueError("Normalized native JSON exceeds the unchanged parser budget")
            destination.write(output)
    if quoted:
        raise ValueError("Native JSON normalization found an unterminated string")
    decode_native_json(target.read_bytes())


def normalize_native_netlist(directory: Path) -> None:
    """Retain original native JSON and produce an exactly token-preserving compact copy.

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
    _compact(source, directory / _TEMPORARY)
    if _digest(source) != before:
        raise ValueError("The original native JSON changed during normalization")
    source.rename(directory / _RAW)
    (directory / _TEMPORARY).rename(source)
    (directory / _RECORD).write_text(
        json.dumps(_record(directory), sort_keys=True, indent=2) + "\n", encoding="utf-8"
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
    if value != _record(directory):
        raise ValueError("The original or compact native JSON differs from its retained record")
    return (_RAW, _RECORD)
