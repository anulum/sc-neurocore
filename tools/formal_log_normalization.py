# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — External formal receipt consistency checks

"""Replace reviewed identity literals while preserving every other log byte."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import sys

from tools.formal_evidence_validation import confined_file, digest, object_fields, require

NORMALIZER_ID = "sc-neurocore-native-log-identity-v1"
PLACEHOLDERS = {
    "workspace": b"<FORMAL_WORKSPACE>",
    "unit": b"<FORMAL_UNIT>",
    "hostname": b"<FORMAL_HOST>",
    "username": b"<FORMAL_USER>",
}


def normalize_native_log(raw: bytes, identities: Mapping[str, str]) -> bytes:
    """Substitute only explicit identity literals; preserve the remaining bytes.

    Parameters
    ----------
    raw : bytes
        Immutable native log, including original line endings.
    identities : mapping of str to str
        Actual workspace, unit, hostname and username. Keep these literals in
        private custody; independent reviewers verify them against execution.

    Returns
    -------
    bytes
        Public technical log with fixed versioned identity placeholders.

    Raises
    ------
    ValueError
        If identities are malformed, placeholders already exist, or the exact
        inverse cannot reproduce the original native bytes.
    """
    require(set(identities) == set(PLACEHOLDERS), "Incomplete identity substitutions.")
    workspace = identities["workspace"]
    require(
        Path(workspace).is_absolute()
        and len(Path(workspace).parts) >= 3
        and not any(character.isspace() for character in workspace),
        "Workspace must be an absolute identity path without whitespace.",
    )
    for name, pattern in (
        ("unit", r"scn-formal-[a-zA-Z0-9_-]+\.service"),
        ("hostname", r"[a-zA-Z0-9][a-zA-Z0-9.-]*"),
        ("username", r"[a-z_][a-z0-9_-]*"),
    ):
        require(re.fullmatch(pattern, identities[name]) is not None, f"Invalid {name} identity.")
    require(
        all(token not in raw for token in PLACEHOLDERS.values()),
        "Raw log already contains normalization placeholders.",
    )
    substitutions = [
        (identities[name].encode("utf-8"), token) for name, token in PLACEHOLDERS.items()
    ]
    require(
        not any(
            literal in token for literal, _ in substitutions for token in PLACEHOLDERS.values()
        ),
        "Identity collides with fixed normalization placeholders.",
    )
    normalized = raw
    for literal, token in substitutions:
        normalized = normalized.replace(literal, token)
    restored = normalized
    for literal, token in reversed(substitutions):
        restored = restored.replace(token, literal)
    require(restored == raw, "Normalization changed bytes outside reversible identity ranges.")
    return normalized


def validate_public_log(repository: Path, value: object) -> Path:
    """Validate distinct raw custody and normalized public log identities.

    Parameters
    ----------
    repository : Path
        Checkout containing the public technical log.
    value : object
        Receipt log metadata, including both hashes and the normalizer binding.

    Returns
    -------
    Path
        Actual normalized log whose bytes match its own digest.

    Raises
    ------
    ValueError
        If normalization identity, custody digest or public bytes differ.
    """
    log = object_fields(value)
    require(log.get("normalizer_id") == NORMALIZER_ID, "Unknown native log normalizer.")
    require(
        log.get("normalizer_sha256") == digest(Path(__file__)),
        "Native log normalizer source differs.",
    )
    raw_digest = log.get("raw_sha256")
    require(
        isinstance(raw_digest, str) and re.fullmatch(r"[0-9a-f]{64}", raw_digest) is not None,
        "Raw native log custody digest is missing or invalid.",
    )
    path = confined_file(repository, log.get("path"))
    require(digest(path) == log.get("normalized_sha256"), "Normalized native log digest differs.")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Write a normalized technical log and print its distinct custody digests.

    Parameters
    ----------
    argv : sequence of str, optional
        CLI arguments; process arguments are used when omitted.

    Returns
    -------
    int
        Zero for reversible normalization, two for a refused input.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    for name in PLACEHOLDERS:
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    try:
        require(
            args.raw.resolve() != args.output.resolve(), "Raw custody file cannot be overwritten."
        )
        raw = args.raw.read_bytes()
        normalized = normalize_native_log(raw, {name: getattr(args, name) for name in PLACEHOLDERS})
        with args.output.open("xb") as stream:
            stream.write(normalized)
        print(
            json.dumps(
                {
                    "normalizer_id": NORMALIZER_ID,
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
                }
            )
        )
    except (ValueError, OSError) as error:
        print(f"Native log normalization refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
