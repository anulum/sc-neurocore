#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust model field documentation from the model descriptors

"""Document Rust neuron model fields from the neuron model descriptors.

Every descriptor under ``src/sc_neurocore/neurons/model_descriptors`` states
what each parameter, and some state variables, of one model mean and in which
unit. The engine's Rust struct of the same name carries the same fields. This
tool writes that meaning and unit as the doc comment of each such public field
that has no documentation yet, so both languages describe a field from one
source. Existing documentation is never replaced.

``--check`` writes nothing, lists the fields that would be documented and
fails when there are any.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Descriptor directory, relative to the repository root.
DESCRIPTOR_DIRECTORY = Path("src/sc_neurocore/neurons/model_descriptors")
#: Rust source tree of the engine crate, relative to the repository root.
ENGINE_SOURCE = Path("engine/src")
#: Descriptor sections whose entries describe struct fields, in lookup order.
FIELD_SECTIONS = ("parameters", "state")

_STRUCT = re.compile(r"^(\s*)pub struct (\w+)\b[^;{]*\{\s*$")
_FIELD = re.compile(r"^(\s*)pub (\w+)\s*:")
_MARKDOWN = re.compile(r"([\[\]*])")


class FieldDocError(ValueError):
    """A descriptor or Rust source cannot be read as this tool requires."""


@dataclass(frozen=True)
class FieldDoc:
    """One undocumented public field and the doc comment line it receives.

    ``line`` is the field's 1-based line; ``insert_at`` is the 1-based line
    the comment takes, above the field's attributes when it has any.
    """

    path: str
    line: int
    insert_at: int
    struct: str
    field: str
    comment: str


def doc_text(meaning: str, unit: str | None) -> str:
    """Render a descriptor meaning and unit as one rustdoc sentence.

    Parameters
    ----------
    meaning : str
        Descriptor ``meaning`` of the field.
    unit : str or None
        Descriptor ``unit``; ``"1"`` is written as dimensionless, and no
        unit is written when it is absent or empty.

    Returns
    -------
    str
        The sentence, capitalised, with Markdown link and emphasis characters
        escaped so rustdoc shows them literally.

    Raises
    ------
    FieldDocError
        If the meaning is empty.
    """
    text = meaning.strip().rstrip(".").strip()
    if not text:
        raise FieldDocError("A descriptor field meaning must not be empty.")
    text = _MARKDOWN.sub(r"\\\1", text[0].upper() + text[1:])
    if unit is None or not unit.strip():
        return f"{text}."
    shown = "dimensionless" if unit.strip() == "1" else unit.strip()
    return f"{text} ({shown})."


def descriptor_docs(root: Path) -> dict[str, dict[str, str]]:
    """Read the field doc sentences of every model descriptor.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    dict of str to dict of str to str
        For each descriptor ``class_name``, the doc sentence of every field
        whose entry has a meaning. A parameter takes precedence over a state
        variable of the same name.

    Raises
    ------
    FieldDocError
        If a descriptor cannot be parsed, has no class name, or shares its
        class name with another descriptor.
    """
    docs: dict[str, dict[str, str]] = {}
    for path in sorted((root / DESCRIPTOR_DIRECTORY).glob("*.toml")):
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
            raise FieldDocError(f"Cannot read descriptor {path.name}.") from exc
        metadata = data.get("metadata")
        name = metadata.get("class_name") if isinstance(metadata, dict) else None
        if not isinstance(name, str) or not name:
            raise FieldDocError(f"Descriptor {path.name} names no class.")
        if name in docs:
            raise FieldDocError(f"Two descriptors describe the class {name}.")
        fields: dict[str, str] = {}
        for section in reversed(FIELD_SECTIONS):
            entries = data.get(section)
            for field, entry in entries.items() if isinstance(entries, dict) else ():
                if isinstance(entry, dict) and isinstance(entry.get("meaning"), str):
                    unit = entry.get("unit")
                    fields[field] = doc_text(
                        entry["meaning"], unit if isinstance(unit, str) else None
                    )
        docs[name] = fields
    return docs


def _struct_fields(
    lines: list[str], start: int, indent: str
) -> list[tuple[int, int, str, str, bool]]:
    """Return the public fields of the struct opened at ``start``.

    Each item is the field's line index, the index where a doc comment goes
    (above any attributes), the field's indentation, its name and whether it
    is already documented.
    """
    fields: list[tuple[int, int, str, str, bool]] = []
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.strip() == "}" and len(line) - len(line.lstrip()) <= len(indent):
            return fields
        match = _FIELD.match(line)
        if match is None:
            continue
        above = index
        while above > start + 1 and lines[above - 1].strip().startswith("#["):
            above -= 1
        attributes = [item.strip() for item in lines[above:index]]
        documented = lines[above - 1].strip().startswith("///") or any(
            item.startswith("#[doc") for item in attributes
        )
        fields.append((index, above, match.group(1), match.group(2), documented))
    raise FieldDocError("A Rust struct is not closed.")


def plan(root: Path) -> list[FieldDoc]:
    """Find every undocumented public field that a descriptor documents.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    list of FieldDoc
        The fields to document, in source order.

    Raises
    ------
    FieldDocError
        If a descriptor is unreadable, or a described struct is defined more
        than once in the engine source or is not closed.
    """
    docs = descriptor_docs(root)
    planned: list[FieldDoc] = []
    defined: dict[str, str] = {}
    for path in sorted((root / ENGINE_SOURCE).rglob("*.rs")):
        relative = path.relative_to(root).as_posix()
        lines = path.read_text(encoding="utf-8").splitlines()
        for start, line in enumerate(lines):
            match = _STRUCT.match(line)
            if match is None or match.group(2) not in docs:
                continue
            struct = match.group(2)
            if struct in defined:
                raise FieldDocError(
                    f"The described struct {struct} is defined in {defined[struct]} and {relative}."
                )
            defined[struct] = relative
            for index, above, indent, field, documented in _struct_fields(
                lines, start, match.group(1)
            ):
                text = docs[struct].get(field)
                if text is not None and not documented:
                    planned.append(
                        FieldDoc(
                            relative, index + 1, above + 1, struct, field, f"{indent}/// {text}"
                        )
                    )
    return planned


def write(root: Path, planned: Sequence[FieldDoc]) -> None:
    """Insert the planned doc comments into their Rust sources.

    Parameters
    ----------
    root : Path
        Repository root.
    planned : sequence of FieldDoc
        Fields from :func:`plan`; each comment goes above the field's
        attributes, or above the field when it has none.
    """
    by_path: dict[str, list[FieldDoc]] = {}
    for item in planned:
        by_path.setdefault(item.path, []).append(item)
    for relative, items in by_path.items():
        path = root / relative
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for item in sorted(items, key=lambda entry: entry.insert_at, reverse=True):
            lines.insert(item.insert_at - 1, item.comment)
        path.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Write descriptor field documentation, or check that none is missing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--check", action="store_true", help="write nothing; fail if a field would be documented"
    )
    args = parser.parse_args(argv)
    try:
        planned = plan(args.repo)
    except (FieldDocError, OSError, UnicodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.check:
        for item in planned:
            print(f"{item.path}:{item.line}: {item.struct}.{item.field} lacks its descriptor doc")
        print(f"{len(planned)} described Rust fields lack documentation.")
        return 1 if planned else 0
    write(args.repo, planned)
    print(f"Documented {len(planned)} Rust fields from the model descriptors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
