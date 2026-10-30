# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Workspace candidate drafts from canonical catalogue schemas

"""Copy an existing schema into a proposal without inventing author or unit metadata."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from sc_neurocore.neurons.model_identity import resolve_identity, schema_for_class
from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.neurons.universal_dsl import load_schema
from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio.candidate_package import CANDIDATE_SCHEMA_VERSION


class CatalogueCandidateUnavailable(AuthoredRefusal):
    """The requested catalogue model has no bound authoring schema."""


def candidate_from_catalogue(model_name: str) -> dict[str, str]:
    """Return a workspace-local draft retaining the complete canonical model.

    Parameters
    ----------
    model_name : str
        Registered catalogue class, resolved through its own schema identity.

    Returns
    -------
    dict[str, str]
        Candidate JSON text, parent identity and an order-sensitive SHA-256 of
        the compact parent schema. Text preserves declaration order and exact
        integer tokens. Authors and reference tests remain empty; undeclared
        units remain blank for the author to supply before validation.

    Raises
    ------
    CatalogueCandidateUnavailable
        If the class is unknown or has no bound canonical schema.
    RuntimeError
        If an identified source resource cannot be read or encoded. Resource
        faults must not be presented as defects in the author's proposal.
    """
    if model_name not in _CLASS_TO_MODULE:
        raise CatalogueCandidateUnavailable("Select a registered catalogue model.")
    try:
        identity = resolve_identity(model_name)
        if not identity.schema_profiles:
            raise CatalogueCandidateUnavailable(
                "This catalogue model has no canonical authoring schema."
            )
        stem = schema_for_class(model_name)
        model = load_schema(stem)
        profile = model.get("profile", {})
        metadata = model.get("metadata", {})
        science = model.get("science", {})
        if not all(isinstance(section, Mapping) for section in (profile, metadata, science)):
            raise ValueError("Invalid schema metadata")
        declared_units = profile.get("units", {})
        if not isinstance(declared_units, Mapping):
            raise ValueError("Invalid schema units")
        time_unit = profile.get("time_unit", "")
        citation = (
            " ".join(
                str(metadata[field]) for field in ("author", "year", "name") if metadata.get(field)
            )
            or f"SC-NeuroCore catalogue schema {stem}"
        )
        source = {"citation": citation}
        for field in ("doi", "url"):
            if metadata.get(field):
                source[field] = str(metadata[field])
        document: dict[str, Any] = {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "name": model_name[:54] + "Candidate",
            "parent": model_name,
            "model": model,
            "units": {
                "state": {name: declared_units.get(name, "") for name in model.get("state", {})},
                "parameters": {
                    name: declared_units.get(name, "") for name in model.get("parameters", {})
                },
                "current": declared_units.get("I", ""),
                "time": "1" if time_unit == "iteration" else time_unit,
            },
            "source": source,
            "assumptions": science.get("assumptions", []),
            "authors": [],
            "reference_tests": [],
        }
        parent_bytes = json.dumps(
            model, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        candidate_text = json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        candidate_text.encode("utf-8")
    except CatalogueCandidateUnavailable:
        raise
    except (OSError, ValueError, TypeError, LookupError, UnicodeError) as exc:
        raise RuntimeError("The canonical authoring schema could not be read.") from exc
    return {
        "schema_version": "studio.candidate-catalogue-start.v1",
        "parent_model": model_name,
        "parent_schema": stem,
        "parent_schema_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "candidate_text": candidate_text,
    }


__all__ = ["CatalogueCandidateUnavailable", "candidate_from_catalogue"]
