# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Typed evidence references and content digests

"""Typed evidence references parsed from descriptor evidence fields.

Descriptor evidence facets (``validation.evidence``, ``silicon.cosim_evidence``,
``silicon.synth_report`` and their siblings) are free strings. This module
parses one field into typed references and resolves each against the
repository:

* ``test-node`` — ``tests/x.py::Class::test_name`` (a pytest node id);
* ``test-file`` — a ``.py`` file under ``tests/``;
* ``artifact-file`` — any other file (report, receipt, trace, RTL);
* ``inline-config`` — a JSON object embedded in the field;
* ``free-text`` — prose that names no file.

Resolution is a filesystem and AST lookup only; nothing here runs the
referenced command. A reference that names a file or a node that does not
exist is reported as ``missing-file`` or ``missing-node`` rather than dropped,
so a fabricated or stale pointer stays visible to the readiness verifier.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

ReferenceKind = Literal["test-node", "test-file", "artifact-file", "inline-config", "free-text"]
Resolution = Literal["resolved", "missing-file", "missing-node", "unresolvable"]

_NODE_ID = re.compile(r"^(?P<path>[\w./-]+\.py)::(?P<node>[\w.:\[\]=,\-]+)$")
_PATH_LIKE = re.compile(r"^[\w./-]+$")
_SEPARATOR = re.compile(r"\s*;\s*")


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    """One parsed and resolved evidence reference.

    Parameters
    ----------
    raw:
        The reference text exactly as written in the descriptor.
    kind:
        Reference kind; see the module docstring.
    path:
        Repository-relative path the reference names (empty for prose and
        inline configurations).
    node:
        ``Class::test`` node path for ``test-node`` references, else empty.
    resolution:
        ``resolved`` when the named file (and node) exists; ``missing-file`` or
        ``missing-node`` when it does not; ``unresolvable`` for prose and inline
        configurations, which name nothing on disk.
    """

    raw: str
    kind: ReferenceKind
    path: str
    node: str
    resolution: Resolution

    @property
    def is_locatable(self) -> bool:
        """True when the reference names a file that could be checked on disk."""
        return self.kind in {"test-node", "test-file", "artifact-file"}

    @property
    def is_resolved(self) -> bool:
        """True when the named file (and node) exists."""
        return self.resolution == "resolved"

    def to_public_dict(self) -> dict[str, str]:
        """Return a JSON-compatible projection."""
        return {
            "raw": self.raw,
            "kind": self.kind,
            "path": self.path,
            "node": self.node,
            "resolution": self.resolution,
        }


def split_evidence_field(value: str) -> tuple[str, ...]:
    """Split a descriptor evidence field into its reference tokens.

    Tokens are separated by semicolons; surrounding whitespace is dropped and
    empty tokens are ignored.

    Parameters
    ----------
    value:
        Raw evidence field text.

    Returns
    -------
    tuple[str, ...]
        Non-empty reference tokens in field order.
    """
    return tuple(token for token in _SEPARATOR.split(value.strip()) if token)


def classify_reference(token: str) -> tuple[ReferenceKind, str, str]:
    """Classify one reference token without touching the filesystem.

    Parameters
    ----------
    token:
        One stripped token from :func:`split_evidence_field`.

    Returns
    -------
    tuple[ReferenceKind, str, str]
        ``(kind, path, node)``.
    """
    if token.startswith("{"):
        try:
            json.loads(token)
        except ValueError:
            return "free-text", "", ""
        return "inline-config", "", ""
    node_match = _NODE_ID.match(token)
    if node_match is not None:
        return "test-node", node_match.group("path"), node_match.group("node")
    if _PATH_LIKE.match(token) and ("/" in token or "." in token):
        if token.startswith("tests/") and token.endswith(".py"):
            return "test-file", token, ""
        return "artifact-file", token, ""
    return "free-text", "", ""


def resolve_reference(token: str, repo_root: Path) -> EvidenceReference:
    """Classify one token and resolve it against ``repo_root``.

    Parameters
    ----------
    token:
        One stripped token from :func:`split_evidence_field`.
    repo_root:
        Repository root the paths are relative to.

    Returns
    -------
    EvidenceReference
        The typed and resolved reference.
    """
    kind, path, node = classify_reference(token)
    if kind in {"free-text", "inline-config"}:
        return EvidenceReference(token, kind, path, node, "unresolvable")
    target = repo_root / path
    if not target.is_file():
        return EvidenceReference(token, kind, path, node, "missing-file")
    if kind == "test-node" and not node_is_defined(target, node):
        return EvidenceReference(token, kind, path, node, "missing-node")
    return EvidenceReference(token, kind, path, node, "resolved")


def parse_evidence_field(value: str, repo_root: Path) -> tuple[EvidenceReference, ...]:
    """Parse and resolve every reference in one descriptor evidence field.

    Parameters
    ----------
    value:
        Raw evidence field text.
    repo_root:
        Repository root the paths are relative to.

    Returns
    -------
    tuple[EvidenceReference, ...]
        References in field order; empty for an empty field.
    """
    return tuple(resolve_reference(token, repo_root) for token in split_evidence_field(value))


@lru_cache(maxsize=512)
def _definitions(path: str, mtime_ns: int) -> frozenset[str]:
    """Return every ``Class::method`` and top-level definition path in a module."""
    del mtime_ns  # part of the cache key only
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    names: set[str] = set()

    def walk(nodes: Iterable[ast.stmt], prefix: str) -> None:
        for statement in nodes:
            if isinstance(statement, ast.ClassDef):
                names.add(prefix + statement.name)
                walk(statement.body, prefix + statement.name + "::")
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(prefix + statement.name)

    walk(tree.body, "")
    return frozenset(names)


def node_is_defined(test_file: Path, node: str) -> bool:
    """Return whether ``node`` (``Class::test[param]``) is defined in ``test_file``.

    Parameters
    ----------
    test_file:
        Absolute path of the test module.
    node:
        Node path after the ``::`` that follows the file name; a trailing
        parametrisation suffix in square brackets is ignored.

    Returns
    -------
    bool
        ``True`` when a class or function with that path is defined.
    """
    bare = node.split("[", maxsplit=1)[0]
    try:
        definitions = _definitions(str(test_file), test_file.stat().st_mtime_ns)
    except (OSError, SyntaxError):
        return False
    return bare in definitions


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of one file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(paths: Iterable[Path], relative_to: Path) -> str:
    """Return one digest over several files, independent of iteration order.

    Parameters
    ----------
    paths:
        Files to include; each is hashed and listed with its path relative to
        ``relative_to``.
    relative_to:
        Root the listed paths are made relative to.

    Returns
    -------
    str
        SHA-256 hex digest of the sorted lines, each holding one relative path,
        a NUL separator and that file's digest.
    """
    lines = sorted(
        f"{path.resolve().relative_to(relative_to.resolve()).as_posix()}\0{sha256_file(path)}\n"
        for path in paths
    )
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def sha256_canonical_json(payload: object) -> str:
    """Return the SHA-256 hex digest of a canonical JSON rendering."""
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


__all__ = [
    "EvidenceReference",
    "ReferenceKind",
    "Resolution",
    "classify_reference",
    "parse_evidence_field",
    "resolve_reference",
    "sha256_canonical_json",
    "sha256_file",
    "sha256_tree",
    "split_evidence_field",
    "node_is_defined",
]
