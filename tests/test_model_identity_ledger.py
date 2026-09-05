# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Generated identity ledger freshness and public count binding

"""The tracked identity ledger and public counts derive from the registry."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

from sc_neurocore.neurons.model_identity import (
    COUNT_DEFINITION,
    catalogue_counts,
    identity_registry,
    public_fidelity_bindings,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_tool() -> ModuleType:
    tool_path = _repo_root() / "tools" / "model_identity_ledger.py"
    spec = importlib.util.spec_from_file_location("model_identity_ledger", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _page_rows(section: str, next_section: str) -> list[str]:
    page = (_repo_root() / "docs/api/model_fidelity_status.md").read_text(encoding="utf-8")
    body = page.split(section, maxsplit=1)[1].split(next_section, maxsplit=1)[0]
    rows = [
        line
        for line in body.splitlines()
        if line.startswith("| ") and not line.startswith(("| Model", "|---"))
    ]
    return [row.split("|", maxsplit=2)[1].strip() for row in rows]


def test_tracked_ledger_is_current() -> None:
    """The generated ledger equals a fresh render of the live registry."""
    tool = _load_tool()
    assert tool.ledger_problems(_repo_root()) == []


def test_ledger_payload_mirrors_registry_and_counts() -> None:
    """Every registry record and count appears once in the tracked ledger."""
    payload = json.loads(
        (_repo_root() / "docs/_generated/model_identity_ledger.json").read_text(encoding="utf-8")
    )
    registry = identity_registry()
    assert payload["schema"] == "sc-neurocore.model-identity-ledger.v1"
    assert payload["count_definition"] == COUNT_DEFINITION
    assert payload["counts"] == catalogue_counts().to_public_dict()
    rows = {row["class_name"]: row for row in payload["identities"]}
    assert set(rows) == set(registry)
    for name, record in registry.items():
        assert rows[name] == record.to_public_dict()
    assert "timestamp" not in payload
    assert "head" not in payload


def test_public_fidelity_page_rows_match_the_registry_bindings() -> None:
    """Each public table row is bound to one class and the bindings cover the page."""
    bindings = public_fidelity_bindings()
    expected = {
        "polyglot-complete": _page_rows(
            "## Polyglot-complete models",
            "## Runtime-validated models awaiting the complete acceleration chain",
        ),
        "runtime-validated": _page_rows(
            "## Runtime-validated models awaiting the complete acceleration chain",
            "## Runtime-complete compatibility identities awaiting benchmark closure",
        ),
        "compatibility-runtime": _page_rows(
            "## Runtime-complete compatibility identities awaiting benchmark closure",
            "## In progress",
        ),
    }
    for status, labels in expected.items():
        bound = sorted(label for label, bound_status in bindings.values() if bound_status == status)
        assert sorted(labels) == bound, status


def test_public_counts_are_derived_not_typed() -> None:
    """README and the fidelity page state exactly the registry-derived numbers."""
    counts = catalogue_counts()
    page = (_repo_root() / "docs/api/model_fidelity_status.md").read_text(encoding="utf-8")
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    assert f"**{counts.polyglot_complete_source}\npolyglot-complete source models**" in page
    assert f"out of the {counts.source_catalogue}-model source catalogue" in page
    assert f"remaining **{counts.remaining_source}** source" in page
    assert f"do not increment the {counts.source_catalogue}-model source catalogue" in page
    assert "tools/model_identity_ledger.py" in page
    assert (
        f"{counts.polyglot_complete_source} of {counts.source_catalogue} catalogue models" in readme
    )
    stale = re.findall(r"\b\d+ of 155 catalogue models\b", readme)
    assert stale == [] or counts.source_catalogue == 155
