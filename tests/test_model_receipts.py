# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reference receipt resolution tests

"""A catalogue binds a model to a receipt only when it can produce the receipt."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from sc_neurocore.neurons.model_identity import iter_source_catalogue, resolve_identity
from sc_neurocore.neurons.model_receipts import (
    RECEIPT_DIRECTORY,
    load_bound_receipt,
    referenced_receipt_name,
)

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

DESCRIPTORS = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_descriptors"
REFERENCE = "src/sc_neurocore/neurons/reference_receipts/{name}"


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        (REFERENCE.format(name="adex.json"), "adex.json"),
        ('{"n_steps":1000,"current":2.0}', None),
        ("src/sc_neurocore/neurons/reference_trace_data/x.json", None),
        (REFERENCE.format(name="nested/adex.json"), None),
        (REFERENCE.format(name="..\\adex.json"), None),
        (REFERENCE.format(name=".hidden.json"), None),
        (REFERENCE.format(name="adex.toml"), None),
        (REFERENCE.format(name=""), None),
    ],
)
def test_only_a_plain_receipt_file_name_is_referenced(reference: str, expected: str | None) -> None:
    """Paths leaving the receipt directory, or naming no JSON receipt, bind nothing."""
    assert referenced_receipt_name(reference) == expected


def test_a_receipt_naming_the_model_or_one_of_its_profiles_binds(tmp_path: Path) -> None:
    """The class itself and a ``Class.profile`` source identity both bind."""
    (tmp_path / "plain.json").write_text(json.dumps({"model": "ExpIFNeuron"}), encoding="utf-8")
    (tmp_path / "profile.json").write_text(
        json.dumps({"model": "ExpIFNeuron.fourcaud_trocme_2003"}), encoding="utf-8"
    )

    for name in ("plain.json", "profile.json"):
        receipt = load_bound_receipt("ExpIFNeuron", REFERENCE.format(name=name), directory=tmp_path)
        assert receipt is not None and str(receipt["model"]).startswith("ExpIFNeuron")


@pytest.mark.parametrize(
    ("content", "why"),
    [
        (None, "absent"),
        ("{not json", "malformed"),
        ("[1, 2]", "not an object"),
        ('{"model": 7}', "model not a string"),
        ('{"equation_origin": {}}', "no model"),
        ('{"model": "ExpIFNeuronVariant"}', "another model sharing a prefix"),
        ('{"model": "AdExNeuron"}', "another model"),
    ],
)
def test_a_receipt_that_cannot_be_produced_or_names_another_model_binds_nothing(
    tmp_path: Path, content: str | None, why: str
) -> None:
    """A reference string alone never makes a model receipt-bound."""
    if content is not None:
        (tmp_path / "receipt.json").write_text(content, encoding="utf-8")
    assert (
        load_bound_receipt("ExpIFNeuron", REFERENCE.format(name="receipt.json"), directory=tmp_path)
        is None
    ), why


def test_undecodable_receipt_bytes_bind_nothing(tmp_path: Path) -> None:
    """A receipt that is not UTF-8 text is not read as evidence."""
    (tmp_path / "receipt.json").write_bytes(b"\xff\xfe\x00{")
    assert (
        load_bound_receipt("ExpIFNeuron", REFERENCE.format(name="receipt.json"), directory=tmp_path)
        is None
    )


def test_every_receipt_bound_identity_binds_a_shipped_receipt() -> None:
    """Each receipt-bound catalogue identity resolves its receipt in this package."""
    bound = [record for record in iter_source_catalogue() if record.revalidation == "receipt-bound"]
    assert bound
    for record in bound:
        descriptor = tomllib.loads((DESCRIPTORS / f"{record.class_name}.toml").read_text("utf-8"))
        reference = str(descriptor["reproducibility"]["reference_config"])
        receipt = load_bound_receipt(record.class_name, reference)
        assert receipt is not None, record.class_name
        name = referenced_receipt_name(reference)
        assert name is not None and (RECEIPT_DIRECTORY / name).is_file()


def test_every_descriptor_receipt_reference_resolves_to_its_own_model() -> None:
    """No descriptor points at a missing receipt or at another model's receipt."""
    references = 0
    for path in sorted(DESCRIPTORS.glob("*.toml")):
        descriptor = tomllib.loads(path.read_text("utf-8"))
        reference = str(descriptor.get("reproducibility", {}).get("reference_config", "") or "")
        if referenced_receipt_name(reference) is None:
            continue
        references += 1
        assert load_bound_receipt(path.stem, reference) is not None, path.stem
    assert references >= 20


def test_a_complete_identity_without_a_receipt_reference_is_not_revalidated() -> None:
    """Promotion without a receipt stays explicitly not revalidated."""
    unbound = [
        record
        for record in iter_source_catalogue()
        if record.public_status == "polyglot-complete" and record.revalidation != "receipt-bound"
    ]
    assert unbound
    assert {resolve_identity(record.class_name).revalidation for record in unbound} == {
        "not-revalidated"
    }
