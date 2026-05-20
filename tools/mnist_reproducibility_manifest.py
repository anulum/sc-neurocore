#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MNIST accuracy reproducibility manifest

"""Generate or verify the committed ConvSpikingNet MNIST evidence manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = Path("examples/mnist_conv_train/results/mnist_training.log")
DEFAULT_CHECKPOINT = Path("examples/mnist_conv_train/results/conv_spiking_net_best.pt")
DEFAULT_OUTPUT = Path("benchmarks/results/mnist_conv_accuracy_reproducibility.json")
EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<epochs>\d+).*?"
    r"train=(?P<train>[0-9.]+)%.*?"
    r"test=(?P<test>[0-9.]+)%.*?"
    r"best=(?P<best>[0-9.]+)%.*?"
    r"lr=(?P<lr>[0-9.]+).*?"
    r"\|\s+(?P<seconds>[0-9.]+)s"
)
PARAM_RE = re.compile(
    r"ConvSpikingNet\s+\|\s+(?P<params>[0-9,]+)\s+params\s+\|\s+"
    r"T=(?P<timesteps>\d+)\s+\|\s+device=(?P<device>\S+)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _parse_training_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    param_match = PARAM_RE.search(text)
    if param_match is None:
        raise ValueError(f"missing ConvSpikingNet parameter header in {path}")

    epochs: list[dict[str, Any]] = []
    for match in EPOCH_RE.finditer(text):
        epochs.append(
            {
                "epoch": int(match.group("epoch")),
                "train_accuracy_percent": float(match.group("train")),
                "test_accuracy_percent": float(match.group("test")),
                "best_test_accuracy_percent": float(match.group("best")),
                "learning_rate": float(match.group("lr")),
                "epoch_wall_time_s": float(match.group("seconds")),
            }
        )
    if not epochs:
        raise ValueError(f"no epoch rows found in {path}")

    best = max(epochs, key=lambda row: row["best_test_accuracy_percent"])
    return {
        "n_parameters": int(param_match.group("params").replace(",", "")),
        "timesteps": int(param_match.group("timesteps")),
        "device": param_match.group("device"),
        "epochs": epochs,
        "best_epoch": int(best["epoch"]),
        "best_test_accuracy_percent": float(best["best_test_accuracy_percent"]),
        "final_epoch": int(epochs[-1]["epoch"]),
        "final_test_accuracy_percent": float(epochs[-1]["test_accuracy_percent"]),
        "total_epoch_wall_time_s": round(sum(row["epoch_wall_time_s"] for row in epochs), 1),
    }


def build_manifest(*, training_log: Path, checkpoint: Path) -> dict[str, Any]:
    """Build the deterministic MNIST reproducibility manifest."""
    log_path = training_log if training_log.is_absolute() else REPO_ROOT / training_log
    checkpoint_path = checkpoint if checkpoint.is_absolute() else REPO_ROOT / checkpoint
    if not log_path.is_file():
        raise FileNotFoundError(log_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    parsed = _parse_training_log(log_path)
    return {
        "schema_version": 1,
        "benchmark_id": "mnist_conv_spiking_net_99_49_2026_03_26",
        "claim": "ConvSpikingNet reached 99.49% MNIST test accuracy in the committed training run.",
        "dataset": {
            "name": "MNIST",
            "split": "test",
            "test_examples": 10000,
            "source": "torchvision.datasets.MNIST",
            "normalization": {"mean": [0.1307], "std": [0.3081]},
            "training_augmentation": ["RandomRotation(10)", "RandomAffine(translate=(0.1, 0.1))"],
        },
        "model": {
            "name": "ConvSpikingNet",
            "architecture": (
                "Conv2d(1,32,5)->LIF->AvgPool2d->Conv2d(32,64,5)->LIF->"
                "AvgPool2d->Linear(1024,128)->LIF->Linear(128,10)->LIF"
            ),
            "n_parameters_from_log": parsed["n_parameters"],
            "learn_beta": True,
            "learn_threshold": True,
            "surrogate": "fast_sigmoid",
            "readout": "membrane accumulation",
        },
        "training": {
            "epochs": parsed["final_epoch"],
            "batch_size": 128,
            "learning_rate": 0.005,
            "optimizer": "AdamW",
            "weight_decay": 0.0001,
            "scheduler": "CosineAnnealingLR",
            "timesteps": parsed["timesteps"],
            "device_from_log": parsed["device"],
            "total_epoch_wall_time_s_from_log": parsed["total_epoch_wall_time_s"],
        },
        "result": {
            "best_test_accuracy_percent": parsed["best_test_accuracy_percent"],
            "best_epoch": parsed["best_epoch"],
            "final_test_accuracy_percent": parsed["final_test_accuracy_percent"],
            "final_epoch": parsed["final_epoch"],
            "claim_boundary": "committed_training_log_and_checkpoint",
        },
        "training_log": {
            "path": _repo_relative(log_path),
            "sha256": _sha256(log_path),
        },
        "checkpoint": {
            "path": _repo_relative(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "format": "PyTorch state_dict",
        },
        "reproduction": {
            "training_command": (
                "python examples/mnist_conv_train.py --epochs 30 --batch-size 128 "
                "--lr 0.005 --timesteps 25 --beta 0.95 --device cuda"
            ),
            "manifest_refresh_command": (
                "python tools/mnist_reproducibility_manifest.py "
                "--output benchmarks/results/mnist_conv_accuracy_reproducibility.json"
            ),
            "requires_torchvision": True,
            "fresh_rerun_required_for_new_release_claims": True,
            "notes": (
                "This manifest validates the committed 99.49% training-run evidence. "
                "A new release claim must attach a fresh rerun manifest instead of "
                "silently reusing this historical artefact."
            ),
        },
    }


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Verify the output file is current")
    args = parser.parse_args()

    payload = build_manifest(training_log=args.training_log, checkpoint=args.checkpoint)
    rendered = _canonical_json(payload)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    if args.check:
        if not output.is_file():
            print(f"missing manifest: {output}", file=sys.stderr)
            return 1
        current = output.read_text(encoding="utf-8")
        if current != rendered:
            print(f"stale manifest: {output}", file=sys.stderr)
            return 1
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
