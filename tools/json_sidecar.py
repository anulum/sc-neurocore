# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ownership sidecars of generated JSON files

"""Write and check the ownership sidecar that every JSON file added since 2026-10-01 carries.

JSON has no comment syntax, so a JSON file's licence and ownership live in the adjacent
``<file>.license`` sidecar: the seven-line project header, then the two copyright records of
the JSON ownership contract. A tool that generates a JSON file writes its sidecar with it and
checks both, so a regenerated file never loses its ownership.

The SPDX tags are joined at run time so that REUSE does not read this module's own source as
declaring a licence expression or a copyright.
"""

from __future__ import annotations

from pathlib import Path

#: The licence line every sidecar carries.
LICENCE = ": ".join(("SPDX-License-Identifier", "AGPL-3.0-or-later"))

#: The two copyright records the JSON ownership contract requires.
COPYRIGHT_RECORDS = (
    ": ".join(
        ("SPDX-FileCopyrightText", "1996-2026 Miroslav Šotek <protoscience@anulum.li> (concepts)")
    ),
    ": ".join(
        ("SPDX-FileCopyrightText", "2020-2026 Miroslav Šotek <protoscience@anulum.li> (code)")
    ),
)

_HEADER_TAIL = (
    "Commercial license available",
    "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
    "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
    "ORCID: 0009-0009-3560-0851",
    "Contact: www.anulum.li | protoscience@anulum.li",
)


def sidecar_path(json_path: Path) -> Path:
    """Return the ``<file>.license`` path next to a JSON file.

    Parameters
    ----------
    json_path : Path
        The JSON file.

    Returns
    -------
    Path
        The sidecar path, ``json_path`` with ``.license`` appended.
    """
    return json_path.with_name(json_path.name + ".license")


def sidecar_text(subject: str) -> str:
    """Return the sidecar text for a JSON file that holds ``subject``.

    Parameters
    ----------
    subject : str
        What the JSON file holds, for the header's title line.

    Returns
    -------
    str
        The seven header lines, a blank line, and the two copyright records, each as a comment.
    """
    title = f"SC-NeuroCore — REUSE sidecar for {subject}"
    header = [LICENCE, *_HEADER_TAIL, title]
    lines = (
        [f"# {line}" for line in header] + [""] + [f"# {record}" for record in COPYRIGHT_RECORDS]
    )
    return "\n".join(lines) + "\n"


def write_sidecar(json_path: Path, subject: str) -> Path:
    """Write the ownership sidecar of a JSON file.

    Parameters
    ----------
    json_path : Path
        The JSON file the sidecar belongs to.
    subject : str
        What the JSON file holds.

    Returns
    -------
    Path
        The sidecar written.
    """
    path = sidecar_path(json_path)
    path.write_text(sidecar_text(subject), encoding="utf-8")
    return path


def sidecar_problems(json_path: Path, subject: str) -> list[str]:
    """Return why a JSON file's sidecar is not the one its generator writes.

    Parameters
    ----------
    json_path : Path
        The JSON file.
    subject : str
        What the JSON file holds.

    Returns
    -------
    list of str
        Empty when the sidecar exists with exactly the generated text; otherwise one problem.
    """
    path = sidecar_path(json_path)
    if not path.is_file():
        return [f"no ownership sidecar {path.name}; run --write"]
    if path.read_text(encoding="utf-8") != sidecar_text(subject):
        return [f"the ownership sidecar {path.name} has drifted; run --write"]
    return []


__all__ = [
    "COPYRIGHT_RECORDS",
    "LICENCE",
    "sidecar_path",
    "sidecar_problems",
    "sidecar_text",
    "write_sidecar",
]
