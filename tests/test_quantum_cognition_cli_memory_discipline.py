# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum cognition CLI memory-discipline contracts

"""Contracts for quantum-cognition CLI SNN stimulus writes."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import cast

from sc_neurocore.quantum_cognition import __main__ as qc_cli
from sc_neurocore.quantum_cognition import gotm_brain

_CANONICAL_KEYS = {
    "content",
    "project",
    "actor",
    "timestamp",
    "entities",
    "kind",
    "source_ref",
}


def _read_single_stimulus(snn_dir: Path) -> dict[str, object]:
    """Return the one stimulus payload written by a focused CLI run."""
    stimuli = sorted(snn_dir.glob("qc_*.json"))
    assert len(stimuli) == 1
    return cast(dict[str, object], json.loads(stimuli[0].read_text(encoding="utf-8")))


def test_cli_default_paths_use_samsung_working_tree() -> None:
    """Default GOTM paths target the Samsung ext4 working checkout."""
    assert qc_cli._DEFAULT_GOTM_PATH == "/media/anulum/GOTM/aaa_God_of_the_Math_Collection"
    assert qc_cli._DEFAULT_SNN_DIR == (
        "/media/anulum/GOTM/aaa_God_of_the_Math_Collection/"
        "04_ARCANE_SAPIENCE/snn_stimuli"
    )
    assert gotm_brain._AGENTIC_SHARED_PATH == (
        "/media/anulum/GOTM/aaa_God_of_the_Math_Collection/agentic-shared"
    )


def test_learn_cli_writes_canonical_snn_stimulus(tmp_path: Path) -> None:
    """The real ``learn`` CLI writes Remanentia-compatible SNN stimuli."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text(
        "# Test repo\n\n"
        "Quantum cognition CLI memory-discipline fixture with enough text "
        "to produce one indexed content chunk.",
        encoding="utf-8",
    )
    snn_dir = tmp_path / "snn"
    state_file = tmp_path / "brain_state.json"

    exit_code = qc_cli.main(
        [
            "learn",
            str(repo),
            "--max-chunks",
            "1",
            "--n-neurons",
            "2",
            "--seed",
            "7",
            "--snn-dir",
            str(snn_dir),
            "--state-file",
            str(state_file),
        ]
    )

    assert exit_code == 0
    payload = _read_single_stimulus(snn_dir)
    assert set(payload) == _CANONICAL_KEYS
    assert payload["project"] == "SC-NEUROCORE"
    assert payload["actor"] == "system"
    assert payload["kind"] == "event"
    assert payload["source_ref"] == "sc_neurocore.quantum_cognition.__main__:_emit_snn_stimulus"
    assert payload["entities"] == ["SC-NEUROCORE", "quantum_cognition"]
    assert isinstance(payload["content"], str)
    assert payload["content"].startswith("QC step 0:")
    assert len(payload["content"]) >= 15
    assert "text" not in payload
    assert "source" not in payload
    timestamp = payload["timestamp"]
    assert isinstance(timestamp, str)
    parsed = datetime.fromisoformat(timestamp)
    assert parsed.tzinfo is not None
    assert state_file.is_file()
