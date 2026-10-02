# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Candidate JSON document admission

"""Encode finite candidate JSON without publishing encoder diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping

from sc_neurocore.refusals import AuthoredRefusal


class CandidateDocumentRefused(AuthoredRefusal):
    """A deliberately authored candidate JSON or Unicode refusal."""


def encode_candidate_document(document: Mapping[str, object]) -> bytes:
    """Encode the candidate's canonical finite JSON representation.

    Parameters
    ----------
    document:
        Candidate fields and metadata, including Unicode attribution.

    Returns
    -------
    bytes
        Sorted, compact UTF-8 JSON, preserving the established valid digest.

    Raises
    ------
    CandidateDocumentRefused
        When values cannot form finite JSON or text contains lone surrogates.
    """
    try:
        encoded = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return encoded.encode("utf-8")
    except UnicodeError as exc:
        raise CandidateDocumentRefused("candidate text must contain valid Unicode") from exc
    except (TypeError, ValueError) as exc:
        raise CandidateDocumentRefused("candidate fields must contain finite JSON values") from exc
