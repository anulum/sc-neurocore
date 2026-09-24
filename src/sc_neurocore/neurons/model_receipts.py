# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reference receipts a model descriptor names

"""Find the reference receipt a descriptor names, inside the installed package.

A descriptor records its source-validation receipt as a repository path under
``neurons/reference_receipts/``. A catalogue that reports a model as bound to a
receipt must be able to produce that receipt: it is read from the package the
code was imported from, never from a neighbouring source checkout, and it must
name the model it is bound to -- the class itself, or one of its source
profiles as ``Class.profile``. A reference that does not resolve to such a
receipt binds nothing, so the model is reported as not revalidated rather than
as validated on the strength of a path string.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

RECEIPT_DIRECTORY = Path(__file__).resolve().parent / "reference_receipts"
"""Where the installed package keeps its reference receipts."""

RECEIPT_MARKER = "neurons/reference_receipts/"
"""The path fragment that marks a descriptor reference as a receipt."""


def referenced_receipt_name(reference: str) -> str | None:
    """Return the receipt file a descriptor reference names.

    Parameters
    ----------
    reference : str
        The descriptor's ``reproducibility.reference_config`` value.

    Returns
    -------
    str or None
        The bare ``*.json`` file name, or ``None`` when the reference names no
        receipt or names one outside the receipt directory.
    """
    if RECEIPT_MARKER not in reference:
        return None
    name = reference.split(RECEIPT_MARKER, 1)[1]
    if not name.endswith(".json") or "/" in name or "\\" in name or name.startswith("."):
        return None
    return name


def load_bound_receipt(
    class_name: str,
    reference: str,
    *,
    directory: Path = RECEIPT_DIRECTORY,
) -> Mapping[str, Any] | None:
    """Load the receipt a descriptor reference binds to ``class_name``.

    Parameters
    ----------
    class_name : str
        The catalogue class the descriptor describes.
    reference : str
        The descriptor's ``reproducibility.reference_config`` value.
    directory : Path
        The receipt directory; the installed package's own by default.

    Returns
    -------
    Mapping or None
        The receipt, or ``None`` when the reference names no receipt, the file
        is absent or unreadable, it is not a JSON object, or it is bound to
        another model.
    """
    name = referenced_receipt_name(reference)
    if name is None:
        return None
    try:
        payload: object = json.loads((directory / name).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    if not isinstance(model, str):
        return None
    if model != class_name and not model.startswith(f"{class_name}."):
        return None
    return payload


__all__ = [
    "RECEIPT_DIRECTORY",
    "RECEIPT_MARKER",
    "load_bound_receipt",
    "referenced_receipt_name",
]
