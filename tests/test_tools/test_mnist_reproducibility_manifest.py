# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MNIST reproducibility manifest tests

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "benchmarks" / "results" / "mnist_conv_accuracy_reproducibility.json"
README = REPO_ROOT / "README.md"
PUBLIC_CLAIM_FILES = (
    README,
    REPO_ROOT / "docs" / "benchmarks" / "training.md",
    REPO_ROOT / "docs" / "benchmarks" / "comparison.md",
    REPO_ROOT / "docs" / "COMPETITIVE_LANDSCAPE.md",
    REPO_ROOT / "docs" / "tutorials" / "28_learning_rules.md",
    REPO_ROOT / "docs" / "tutorials" / "46_neurobench.md",
    REPO_ROOT / "docs" / "guides" / "FOR_RESEARCH_LABS.md",
    REPO_ROOT / "docs" / "architecture" / "COMPONENT_INVENTORY.md",
    REPO_ROOT / "docs" / "reports" / "SC_NEUROCORE_CAPABILITY_REPORT_v3.13.md",
)


def test_mnist_reproducibility_manifest_is_committed_and_claim_safe() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert payload["benchmark_id"] == "mnist_conv_spiking_net_99_49_2026_03_26"
    assert payload["dataset"]["name"] == "MNIST"
    assert payload["result"]["best_test_accuracy_percent"] == 99.49
    assert payload["result"]["best_epoch"] == 30
    assert payload["result"]["claim_boundary"] == "committed_training_log_and_checkpoint"
    assert (
        payload["checkpoint"]["path"]
        == "examples/mnist_conv_train/results/conv_spiking_net_best.pt"
    )
    assert len(payload["checkpoint"]["sha256"]) == 64
    assert payload["training_log"]["path"] == "examples/mnist_conv_train/results/mnist_training.log"
    assert len(payload["training_log"]["sha256"]) == 64
    assert "python examples/mnist_conv_train.py" in payload["reproduction"]["training_command"]
    assert payload["reproduction"]["requires_torchvision"] is True
    assert payload["reproduction"]["fresh_rerun_required_for_new_release_claims"] is True


def test_mnist_reproducibility_manifest_check_mode_matches_committed_file() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/mnist_reproducibility_manifest.py",
            "--check",
            "--output",
            str(MANIFEST),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_public_mnist_claims_cite_reproducibility_manifest() -> None:
    manifest_ref = "`benchmarks/results/mnist_conv_accuracy_reproducibility.json`"

    for path in PUBLIC_CLAIM_FILES:
        text = path.read_text(encoding="utf-8")
        if "99.49" in text:
            assert manifest_ref in text, path
