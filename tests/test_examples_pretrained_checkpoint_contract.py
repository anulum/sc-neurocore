# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pretrained example checkpoint trust-boundary tests

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "examples" / "12_load_pretrained_model.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("scn_pretrained_example", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load pretrained example module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("bad_digest", ["", "abc123", "g" * 64, "F" * 63])
def test_pretrained_example_rejects_invalid_digest_format(bad_digest: str) -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="checkpoint_sha256 must be exactly 64 hexadecimal"):
        module._require_valid_checkpoint_sha256(bad_digest)


def test_pretrained_example_rejects_non_dict_checkpoint_schema() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="checkpoint payload must be a dictionary"):
        module._require_checkpoint_schema(None)


def test_pretrained_example_rejects_missing_state_dict_schema() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="model_state_dict"):
        module._require_checkpoint_schema({"best_accuracy": 0.97})


def test_pretrained_example_rejects_non_numeric_accuracy() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="best_accuracy"):
        module._require_checkpoint_schema(
            {
                "model_state_dict": {"layer.weight": torch.zeros(1, dtype=torch.float32)},
                "best_accuracy": "not-a-number",
            }
        )


def test_pretrained_example_cli_requires_checkpoint_sha256(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'src'}:{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    )
    result = subprocess.run(
        [sys.executable, str(MODULE_PATH)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert "--checkpoint-sha256" in result.stderr
