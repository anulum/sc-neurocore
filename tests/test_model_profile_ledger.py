# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Generated model profile ledger freshness

"""The tracked model profile ledger mirrors the live registry."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from sc_neurocore.neurons.model_profile import METHOD_TABLE, PROFILE_CONTRACT
from sc_neurocore.neurons.profile_registry import profile_inventory, summarise_inventory


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_tool() -> ModuleType:
    tool_path = _repo_root() / "tools" / "model_profile_ledger.py"
    spec = importlib.util.spec_from_file_location("model_profile_ledger", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ledger() -> dict[str, Any]:
    path = _repo_root() / "docs/_generated/model_profile_ledger.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def test_tracked_ledger_is_current() -> None:
    """The generated ledger equals a fresh render of the live registry."""
    tool = _load_tool()
    assert tool.ledger_problems(_repo_root()) == []


def test_ledger_payload_mirrors_the_registry() -> None:
    """Every profile row, the contract and the mapping table in the ledger are the live ones."""
    payload = _ledger()
    assert payload["schema"] == "sc-neurocore.model-profile-ledger.v1"
    assert payload["contract"] == PROFILE_CONTRACT
    assert payload["method_table"] == [dict(row) for row in METHOD_TABLE]
    rows = profile_inventory()
    assert payload["summary"] == summarise_inventory(rows)
    assert payload["profiles"] == [row.to_public_dict() for row in rows]
    assert "timestamp" not in payload
    assert "head" not in payload


def test_summary_cli_prints_the_partition(capsys: pytest.CaptureFixture[str]) -> None:
    """The --summary mode prints the same partition the ledger carries."""
    tool = _load_tool()
    assert tool.main(["--summary"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == _ledger()["summary"]
