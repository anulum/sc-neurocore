# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Facet receipt file ingress and credit refusals

"""Reject malformed receipt files and refuse invalid readiness credit."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.neurons.facet_receipts import (
    RECEIPT_DIR,
    FacetReceiptError,
    credit_problems,
    iter_receipts,
    load_receipt,
    receipt_filename,
    seal_digest,
)

_REAL_RECEIPT = RECEIPT_DIR / "LapicqueNeuron__cosim__20260923T041700Z__lapicque.json"


def _write_variant(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("class_name", None, "non-empty string"),
        ("tool", [], "map strings to strings"),
        ("evidence_refs", "not-a-list", "list of strings"),
        ("counts", {"passed": True}, "must be an integer"),
        ("subjects", [42], "entries must be objects"),
        ("subjects", {}, "subjects must be a list"),
        ("artifacts", {}, "artifacts must be a list"),
        ("counts", [], "counts must be an object"),
    ],
)
def test_receipt_file_rejects_malformed_fields(
    tmp_path: Path, field: str, invalid: object, message: str
) -> None:
    """The package's actual receipt format rejects corrupt file fields."""
    payload: dict[str, Any] = json.loads(_REAL_RECEIPT.read_text(encoding="utf-8"))
    payload[field] = invalid
    with pytest.raises(FacetReceiptError, match=message):
        load_receipt(_write_variant(tmp_path, payload))


def test_receipt_file_must_be_an_object(tmp_path: Path) -> None:
    """A readable JSON scalar cannot enter the readiness verifier."""
    with pytest.raises(FacetReceiptError, match="not a JSON object"):
        load_receipt(_write_variant(tmp_path, ["receipt"]))


def test_seal_rejects_already_sealed_payload() -> None:
    """Sealing an already sealed package receipt cannot hide its old digest."""
    payload = json.loads(_REAL_RECEIPT.read_text(encoding="utf-8"))
    with pytest.raises(FacetReceiptError, match="must not carry receipt_sha256"):
        seal_digest(payload)


def test_credit_rejects_legacy_unknown_and_negative_receipts() -> None:
    """A valid file cannot credit a superseded schema or malformed run counts."""
    receipt = load_receipt(_REAL_RECEIPT)
    legacy = replace(receipt, schema="sc-neurocore.facet-receipt.v1").sealed()
    assert any("legacy receipt" in problem for problem in credit_problems(legacy))
    unknown = replace(receipt, facet="unregistered").sealed()
    assert credit_problems(unknown) == ("unknown facet 'unregistered'",)
    negative = replace(receipt, counts={**receipt.counts, "failed": -1}).sealed()
    assert any("negative check count" in problem for problem in credit_problems(negative))


def test_receipt_inventory_and_name_refuse_absent_or_escaping_inputs(tmp_path: Path) -> None:
    """An absent inventory is empty and an escaping profile has no file name."""
    assert list(iter_receipts(tmp_path / "absent")) == []
    with pytest.raises(FacetReceiptError, match="schema stem"):
        receipt_filename("LapicqueNeuron", "cosim", "2026-09-23T04:17:00Z", profile="../x")
