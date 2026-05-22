# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Tests for Q8.8 cosimulation tool import behaviour

"""Import-contract tests for the Q8.8 cosimulation tool."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cosim_q88_tool_import_preserves_cwd(tmp_path: Path) -> None:
    """Importing the tool must not move callers into the SHD training tree."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path\n"
                "start = Path.cwd()\n"
                "import tools.cosim_q88_vs_pytorch\n"
                "assert Path.cwd() == start, (start, Path.cwd())\n"
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr


def test_cosim_q88_cli_requires_checkpoint_sha256(tmp_path: Path) -> None:
    """The legacy SHD checkpoint path must be digest-gated at the CLI boundary."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/cosim_q88_vs_pytorch.py"),
            "--checkpoint",
            str(tmp_path / "missing.pth"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "--stride",
            "1",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 2
    assert "--checkpoint-sha256" in result.stderr


@pytest.mark.parametrize("bad_digest", ["", "abc123", "g" * 64, "A" * 63])
def test_cosim_q88_rejects_invalid_checkpoint_digest_format(bad_digest: str) -> None:
    """Main boundary must reject malformed checkpoint digests before loading."""
    from tools.cosim_q88_vs_pytorch import _require_valid_checkpoint_sha256

    with pytest.raises(ValueError, match="checkpoint_sha256 must be exactly 64 hexadecimal"):
        _require_valid_checkpoint_sha256(bad_digest)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (None, "checkpoint must be a dictionary payload"),
        ({}, "checkpoint\\['net'\\] must be a non-empty dictionary"),
        ({"net": {}}, "checkpoint\\['net'\\] must be a non-empty dictionary"),
        ({"net": {"w": object()}, "acc": 1.0}, "must be a torch.Tensor"),
        ({"net": {"": object()}, "acc": 1.0}, "keys must be non-empty strings"),
        ({"net": {"w": 1}, "acc": 1.0}, "must be a torch.Tensor"),
        ({"net": {"w": 1}, "acc": "nan"}, "must be a torch.Tensor"),
        ({"net": {"w": 1}}, "must be a torch.Tensor"),
    ],
)
def test_cosim_q88_checkpoint_contract_rejects_malformed_payload(
    payload: object, match: str
) -> None:
    """Checkpoint schema guard must fail closed on malformed payloads."""
    from tools.cosim_q88_vs_pytorch import _require_legacy_checkpoint_contract

    with pytest.raises(ValueError, match=match):
        _require_legacy_checkpoint_contract(payload)


def test_cosim_q88_checkpoint_contract_requires_numeric_finite_acc() -> None:
    """Checkpoint metadata must include finite numeric validation accuracy."""
    import torch

    from tools.cosim_q88_vs_pytorch import _require_legacy_checkpoint_contract

    with pytest.raises(ValueError, match="checkpoint\\['acc'\\] must be numeric"):
        _require_legacy_checkpoint_contract({"net": {"w": torch.tensor([1.0])}, "acc": "bad"})

    with pytest.raises(ValueError, match="checkpoint\\['acc'\\] must be finite"):
        _require_legacy_checkpoint_contract(
            {"net": {"w": torch.tensor([1.0])}, "acc": float("nan")}
        )
