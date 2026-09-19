# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum cognition CLI memory-discipline contracts

"""Contracts for quantum-cognition CLI SNN stimulus writes."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import cast

import pytest

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


def test_no_workstation_path_is_baked_into_the_shipped_package() -> None:
    """A published wheel must not carry one machine's absolute path as a default.

    These defaults used to be the operator's own collection root, asserted here
    literally. That path exists on exactly one machine, so every other
    installation defaulted to nothing while claiming a default. The root now
    comes from ``SC_NEUROCORE_GOTM_ROOT``; what is pinned is the derivation, not
    a location.
    """
    for module in (qc_cli, gotm_brain):
        source = Path(module.__file__ or "").read_text(encoding="utf-8")
        assert "/media/anulum" not in source, module.__name__
        assert "/home/anulum" not in source, module.__name__


def test_collection_paths_derive_from_the_configured_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every collection path is built from the configured root, not written down."""
    monkeypatch.setenv(qc_cli.GOTM_ROOT_ENV_VAR, "/srv/collection")
    reloaded_cli = importlib.reload(qc_cli)
    reloaded_brain = importlib.reload(gotm_brain)
    try:
        assert reloaded_cli._DEFAULT_GOTM_PATH == "/srv/collection"
        assert reloaded_cli._DEFAULT_SNN_DIR == ("/srv/collection/04_ARCANE_SAPIENCE/snn_stimuli")
        assert reloaded_brain._AGENTIC_SHARED_PATH == "/srv/collection/agentic-shared"
    finally:
        monkeypatch.delenv(qc_cli.GOTM_ROOT_ENV_VAR, raising=False)
        importlib.reload(qc_cli)
        importlib.reload(gotm_brain)


def test_learn_refuses_when_no_repository_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no path and no configured root, the CLI refuses instead of guessing."""
    monkeypatch.delenv(qc_cli.GOTM_ROOT_ENV_VAR, raising=False)
    reloaded = importlib.reload(qc_cli)
    try:
        args = argparse.Namespace(
            repo_path=None,
            state_file=None,
            model=None,
            n_neurons=4,
            seed=1,
            max_chunks=1,
            snn_dir=None,
        )
        assert reloaded.cmd_learn(args) == 2
    finally:
        importlib.reload(qc_cli)


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
    assert isinstance(timestamp, int)
    assert timestamp > 0
    assert state_file.is_file()
