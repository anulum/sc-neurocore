# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio cross-runtime evidence seal

"""One digest that means the same thing on the server and in the browser.

Every Studio evidence lane sealed its payloads with ``json.dumps(...,
sort_keys=True)`` and the digest of that text. That text is Python's, not
JSON's. Measured on the real surface before this module existed: a model run of
``AdExNeuron`` recorded ``result_sha256``
``771d8e51…``; the identical payload, after the round trip every exported
artefact makes through the operator's browser, re-sealed to ``813d3005…``.
The values were unchanged — Python compared the two payloads equal — but
``1.0`` had come back as ``1``, ``-70.0`` as ``-70`` and ``1e-07`` as
``1e-7``. No verifier could exist: checking a recorded seal against a
browser-supplied payload would have failed for every honest run.

The seal here is defined over *values*, not over one runtime's rendering of
them. A number is written in one normal form that both runtimes reach from the
same double, keys are ordered by code point rather than by UTF-16 unit, strings
carry the minimal RFC 8259 escapes, and anything that cannot survive the trip
intact — a non-finite float, an integer no double holds exactly, an unpaired
surrogate — is refused instead of silently altered.

``studio/frontend/src/evidenceSeal.ts`` is the byte-identical counterpart.
Neither runtime can perform the other's JSON round trip, so parity is carried
by two committed corpora that both read —
``studio/frontend/src/evidenceSealVectors.json`` (the cases a person would
write down) and ``evidenceSealRandomVectors.json`` (1024 doubles drawn as bit
patterns, written by ``tools/generate_evidence_seal_vectors.py``). Each side
parses the same JSON with its own parser and must reach the same canonical
text. ``tests/test_studio_evidence_seal.py`` holds this side and the freshness
of the corpora; ``evidenceSeal.test.ts`` holds the other and additionally puts
every vector through a real ``JSON.parse(JSON.stringify(...))`` before sealing
it, which is the browser round trip itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math

#: Contract version of the canonical form. A change to the normal form is a
#: change of digest for the same value, so it takes a new version.
EVIDENCE_SEAL_SCHEMA_VERSION = "studio.evidence-seal.v1"

#: Digest algorithm named in every receipt that carries a seal.
EVIDENCE_SEAL_ALGORITHM = "sha256"


_SHORT_ESCAPES = {
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
    '"': '\\"',
    "\\": "\\\\",
}


class EvidenceSealError(ValueError):
    """Raised when a value cannot be sealed identically in both runtimes."""


def seal_sha256(value: object) -> str:
    """Return the SHA-256 digest of the canonical form of ``value``.

    Parameters
    ----------
    value : object
        Any JSON-shaped value: ``None``, ``bool``, ``int``, ``float``, ``str``,
        a mapping with string keys, or a sequence of the same.

    Returns
    -------
    str
        Lowercase 64-character hexadecimal digest.

    Raises
    ------
    EvidenceSealError
        If the value contains anything that would not survive a JSON round
        trip through the browser unchanged.
    """
    return hashlib.sha256(canonical_seal_text(value).encode("utf-8")).hexdigest()


def canonical_seal_text(value: object) -> str:
    """Return the canonical JSON text used as the seal input.

    Parameters
    ----------
    value : object
        The value to encode.

    Returns
    -------
    str
        Canonical JSON: object keys in code-point order, no insignificant
        whitespace, numbers in the shared normal form, minimal string escapes.

    Raises
    ------
    EvidenceSealError
        If the value is not JSON-shaped, carries a non-finite float, an integer
        no double holds exactly, or an unpaired surrogate.
    """
    parts: list[str] = []
    _encode(value, parts)
    return "".join(parts)


def _encode(value: object, parts: list[str]) -> None:
    if value is None:
        parts.append("null")
        return
    if isinstance(value, bool):
        parts.append("true" if value else "false")
        return
    if isinstance(value, int):
        parts.append(_canonical_integer(value))
        return
    if isinstance(value, float):
        parts.append(_canonical_float(value))
        return
    if isinstance(value, str):
        parts.append(_canonical_string(value))
        return
    if isinstance(value, Mapping):
        _encode_mapping(value, parts)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        _encode_sequence(value, parts)
        return
    raise EvidenceSealError(f"{type(value).__name__} is not a sealable JSON value.")


def _encode_mapping(value: Mapping[object, object], parts: list[str]) -> None:
    keys: list[str] = []
    for key in value:
        if not isinstance(key, str):
            raise EvidenceSealError("A sealable object key must be a string.")
        keys.append(key)
    parts.append("{")
    for index, key in enumerate(sorted(keys, key=_code_points)):
        if index:
            parts.append(",")
        parts.append(_canonical_string(key))
        parts.append(":")
        _encode(value[key], parts)
    parts.append("}")


def _encode_sequence(value: Sequence[object], parts: list[str]) -> None:
    parts.append("[")
    for index, item in enumerate(value):
        if index:
            parts.append(",")
        _encode(item, parts)
    parts.append("]")


def _code_points(value: str) -> tuple[int, ...]:
    """Return the code points of ``value`` for runtime-independent ordering.

    Python orders strings by code point and JavaScript by UTF-16 code unit; the
    two disagree above the basic multilingual plane. Ordering by this key makes
    both runtimes agree.
    """
    return tuple(ord(character) for character in value)


def _canonical_integer(value: int) -> str:
    """Return the token for an integer, refusing one no double holds exactly.

    A JSON number reaches the browser as a double and comes back as one. An
    integer the conversion changes would be sealed under a value it does not
    keep, so it is refused; the magnitude alone does not decide this, because a
    large integer with enough factors of two survives intact.
    """
    try:
        exact = int(float(value)) == value
    except OverflowError:
        exact = False
    if not exact:
        raise EvidenceSealError(
            f"Integer {value} is not carried exactly by a JSON number; "
            "seal it as a string if it must be preserved."
        )
    return str(value)


def _canonical_float(value: float) -> str:
    """Return the normal form of a finite double.

    An integral double is written as the integer it equals, because that is
    what the browser sends back for it. Any other double is written in
    scientific form built from its shortest round-tripping digits, which both
    runtimes produce identically from the same double.
    """
    if not math.isfinite(value):
        raise EvidenceSealError(f"{value!r} is not a sealable JSON number.")
    if value.is_integer():
        return str(int(value))
    significand, _, exponent_text = repr(abs(value)).partition("e")
    integer_text, _, fraction = significand.partition(".")
    # The shortest round-tripping form never ends in a redundant zero, in either
    # runtime, so only the leading zeros of a value below one are dropped.
    digits = f"{integer_text}{fraction}".lstrip("0")
    exponent = int(exponent_text or 0) - len(fraction) + len(digits) - 1
    mantissa = f"{digits[0]}.{digits[1:]}" if len(digits) > 1 else digits
    return f"{'-' if value < 0 else ''}{mantissa}e{exponent}"


def _canonical_string(value: str) -> str:
    parts = ['"']
    for character in value:
        escape = _SHORT_ESCAPES.get(character)
        if escape is not None:
            parts.append(escape)
            continue
        code_point = ord(character)
        if code_point < 0x20:
            parts.append(f"\\u{code_point:04x}")
            continue
        if 0xD800 <= code_point <= 0xDFFF:
            raise EvidenceSealError("A sealable string must not contain an unpaired surrogate.")
        parts.append(character)
    parts.append('"')
    return "".join(parts)


__all__ = [
    "EVIDENCE_SEAL_ALGORITHM",
    "EVIDENCE_SEAL_SCHEMA_VERSION",
    "EvidenceSealError",
    "canonical_seal_text",
    "seal_sha256",
]
