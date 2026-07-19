# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — gate: every citeable descriptor DOI is verified in the committed ledger

"""Offline gate on the integrity of every neuron-descriptor provenance DOI.

Fabricated, mistyped, or misattributed DOIs have repeatedly reached the model
catalogue via generated descriptors (a non-existent "Kilinc & Bhatt 2023", the
recurring phantom "Bhatt" co-author, digit-transposed DOIs, and one DOI that
resolved to an unrelated paper on endotoxin shock). These tests turn that into a
hard, permanent gate: every descriptor that claims a DOI must have that DOI
recorded as ``verified`` in ``tools/provenance/doi_ledger.json`` — the ledger the
online verifier (``tools/provenance/verify_dois.py``) writes only after resolving
the DOI at its registry (Crossref, or DataCite for arXiv) and confirming the
first author and year agree with the descriptor.

The gate runs fully offline, so it is fast and deterministic in CI. A newly
added or edited DOI that has not been re-verified simply is absent from the
ledger (or present but not ``verified``) and the gate fails, forcing the author
to run the verifier — which in turn fails on a fabricated DOI. To refresh after a
legitimate change: ``python tools/provenance/verify_dois.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ImportError:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPO_ROOT / "tools" / "provenance" / "doi_ledger.json"
DESCRIPTOR_DIR = REPO_ROOT / "src" / "sc_neurocore" / "neurons" / "model_descriptors"

sys.path.insert(0, str(REPO_ROOT / "src"))

from sc_neurocore.neurons.model_catalogue import load_descriptor  # noqa: E402


def _descriptor_dois() -> list[tuple[str, str]]:
    """Return every primary or named secondary provenance DOI in descriptors."""
    rows: list[tuple[str, str]] = []
    for toml_path in sorted(DESCRIPTOR_DIR.glob("*.toml")):
        descriptor = load_descriptor(toml_path.stem)
        if descriptor.provenance.doi:
            rows.append((toml_path.stem, descriptor.provenance.doi))
        payload = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        raw_provenance = payload.get("provenance", {})
        if not isinstance(raw_provenance, dict):
            continue
        for key, value in sorted(raw_provenance.items()):
            if key != "doi" and not key.endswith("_doi"):
                continue
            if isinstance(value, str) and value and value != descriptor.provenance.doi:
                rows.append((f"{toml_path.stem}:{key}", value))
    return rows


def _ledger() -> dict[str, dict[str, object]]:
    assert LEDGER_PATH.exists(), (
        f"provenance DOI ledger missing at {LEDGER_PATH}; run `python tools/provenance/verify_dois.py`"
    )
    data: dict[str, dict[str, object]] = json.loads(LEDGER_PATH.read_text())
    return data


DESCRIPTOR_DOIS = _descriptor_dois()


def test_catalogue_has_citeable_dois() -> None:
    """Guard against a silently empty scan hiding every other assertion."""
    assert len(DESCRIPTOR_DOIS) >= 100, (
        f"only {len(DESCRIPTOR_DOIS)} descriptor DOIs found; expected the full citeable catalogue"
    )


@pytest.mark.parametrize(("model", "doi"), DESCRIPTOR_DOIS, ids=lambda pair: str(pair))
def test_every_descriptor_doi_is_verified(model: str, doi: str) -> None:
    """Each descriptor DOI must resolve at its registry with a matching author and year."""
    ledger = _ledger()
    assert doi in ledger, (
        f"{model}: DOI {doi} is not in the verification ledger — run the verifier; "
        f"a DOI absent from the ledger has never been resolved at its registry"
    )
    entry = ledger[doi]
    assert entry["resolves"] is True, (
        f"{model}: DOI {doi} does not resolve at {entry['registry']} (fabricated or mistyped)"
    )
    # A declared translation/reprint deliberately keeps the original work's author
    # and year while the DOI points to a later translation, so the literal
    # author/year comparison is expected to differ; resolution + verification still
    # apply (the DOI must be a real, resolvable paper).
    if not entry.get("translation"):
        assert entry["author_match"] is True, (
            f"{model}: DOI {doi} resolves but its first author "
            f"({entry.get('registry_first_author')!r}) does not match the descriptor — "
            f"the DOI likely points to a different paper"
        )
        assert entry["year_match"] is True, (
            f"{model}: DOI {doi} resolves but its year ({entry.get('registry_year')}) "
            f"is more than a year from the descriptor's"
        )
    assert entry["verified"] is True, f"{model}: DOI {doi} is not marked verified in the ledger"


def test_ledger_has_no_orphan_entries() -> None:
    """Every ledger DOI is claimed by some descriptor (keeps the ledger from accreting dead rows)."""
    claimed = {doi for _model, doi in DESCRIPTOR_DOIS}
    orphans = set(_ledger()) - claimed
    assert not orphans, f"ledger has {len(orphans)} DOI(s) no descriptor claims: {sorted(orphans)}"
