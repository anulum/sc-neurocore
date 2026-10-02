# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public candidate validation and JSON admission

"""Verify public candidate refusals against real schemas and JSON encoding."""

from __future__ import annotations

import hashlib
import json

import pytest

from sc_neurocore.studio.candidate_document import CandidateDocumentRefused
from sc_neurocore.studio.candidate_package import candidate_sha256, validate_candidate
from tests.studio_candidate_support import adex_candidate


@pytest.mark.parametrize("value", [chr(0xD800), float("nan"), object()])
def test_public_digest_refuses_unencodable_candidate_metadata(value: object) -> None:
    """Raise a source-authored refusal for text and non-JSON metadata failures."""
    document = adex_candidate()
    document["model"]["metadata"]["extra"] = value
    with pytest.raises(CandidateDocumentRefused) as refused:
        candidate_sha256(document)
    assert isinstance(refused.value, ValueError)
    assert str(refused.value) in {
        "candidate text must contain valid Unicode",
        "candidate fields must contain finite JSON values",
    }
    validation = validate_candidate(document)
    assert not validation.valid
    assert validation.candidate_sha256 is None
    assert validation.diagnostics[-1].message == str(refused.value)


def test_public_digest_preserves_the_existing_finite_unicode_representation() -> None:
    """Keep the established canonical digest for a valid attributed candidate."""
    document = adex_candidate()
    document["source"]["citation"] += " — Šotek"
    original_bytes = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert candidate_sha256(document) == hashlib.sha256(original_bytes).hexdigest()
    assert validate_candidate(document).valid


def test_candidate_validation_retains_the_deliberate_equation_explanation() -> None:
    """Preserve the sandbox's authored reason through public validation."""
    document = adex_candidate()
    document["model"]["dynamics"]["v"] = "__import__('os')"
    validation = validate_candidate(document)
    assert not validation.valid
    assert validation.diagnostics[0].location == "/model/dynamics/v"
    assert validation.diagnostics[0].message == (
        "Blocked function '__import__' in equation: \"__import__('os')\""
    )


def test_non_string_parent_is_a_located_candidate_refusal() -> None:
    """Reject an actual JSON object parent before membership hashing fails."""
    document = adex_candidate()
    document["parent"] = {"name": "AdExNeuron"}
    validation = validate_candidate(document)
    assert not validation.valid
    assert validation.diagnostics[0].location == "/parent"
    assert validation.diagnostics[0].message == (
        "parent {'name': 'AdExNeuron'} is not a catalogue model"
    )


def test_fractional_unit_powers_keep_the_existing_unit_parser_semantics() -> None:
    """Retain valid fractional units while refusing complex unit exponents."""
    document = adex_candidate()
    document["units"]["state"]["v"] = "mV**0.5"
    assert validate_candidate(document).valid
