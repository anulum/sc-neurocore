# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Load pretrained ConvSpikingNet and classify MNIST digits

"""Load pretrained ConvSpikingNet and classify MNIST digits.

Demonstrates:
1. Loading pretrained weights from weights/ directory
2. Running inference with surrogate-gradient-trained SNN
3. Extracting SC-normalized weights for bitstream deployment

Requires: torch, torchvision
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    import torch
except ImportError:
    raise SystemExit("pip install torch")

from sc_neurocore.security.checkpoint_loading import safe_load_checkpoint
from sc_neurocore.training.snn_modules import ConvSpikingNet

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _require_valid_checkpoint_sha256(digest: str) -> str:
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError("checkpoint_sha256 must be exactly 64 hexadecimal characters")
    return digest


def _require_checkpoint_schema(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint['model_state_dict'] must be a non-empty dictionary")
    best_accuracy = payload.get("best_accuracy")
    if not isinstance(best_accuracy, (int, float)):
        raise ValueError("checkpoint['best_accuracy'] must be numeric")
    if not torch.isfinite(torch.tensor(float(best_accuracy), dtype=torch.float64)):
        raise ValueError("checkpoint['best_accuracy'] must be finite")
    if not 0.0 <= float(best_accuracy) <= 1.0:
        raise ValueError("checkpoint['best_accuracy'] must be within [0, 1]")

    n_params = payload.get("n_params")
    if not isinstance(n_params, int) or n_params <= 0:
        raise ValueError("checkpoint['n_params'] must be a positive integer")

    sc_weights = payload.get("sc_weights")
    if not isinstance(sc_weights, list):
        raise ValueError("checkpoint['sc_weights'] must be a list")
    for idx, item in enumerate(sc_weights):
        if not isinstance(item, torch.Tensor):
            raise ValueError(f"checkpoint['sc_weights'][{idx}] must be a torch.Tensor")
    return payload


def load_pretrained(weights_dir: Path | None = None, *, checkpoint_sha256: str):
    if weights_dir is None:
        weights_dir = Path(__file__).resolve().parent.parent / "weights"
    ckpt_path = weights_dir / "conv_spiking_net_mnist.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found. Run: python tools/train_pretrained_mnist.py"
        )
    trusted_digest = _require_valid_checkpoint_sha256(checkpoint_sha256)
    checkpoint = safe_load_checkpoint(
        ckpt_path,
        trusted_sha256={ckpt_path.name: trusted_digest},
        map_location="cpu",
    )
    checkpoint = _require_checkpoint_schema(checkpoint)
    model = ConvSpikingNet(n_output=10)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Directory containing conv_spiking_net_mnist.pt",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        help="Expected SHA-256 digest for conv_spiking_net_mnist.pt",
    )
    args = parser.parse_args(argv)

    try:
        from torchvision import datasets, transforms
    except ImportError:
        raise SystemExit("pip install torchvision")
    model, ckpt = load_pretrained(args.weights_dir, checkpoint_sha256=args.checkpoint_sha256)
    print(f"Loaded ConvSpikingNet (test accuracy: {ckpt['best_accuracy']:.1%})")
    print(f"  Parameters: {ckpt['n_params']:,}")
    print(f"  SC weight matrices: {len(ckpt['sc_weights'])}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

    correct = 0
    n_test = 100
    T = 25
    with torch.no_grad():
        for i in range(n_test):
            img, label = test_ds[i]
            x = img.unsqueeze(0).unsqueeze(0).expand(T, -1, -1, -1, -1)
            spike_counts, _ = model(x)
            pred = spike_counts.argmax(1).item()
            correct += pred == label

    print(
        f"\nInference on {n_test} test images: {correct}/{n_test} correct ({correct / n_test:.0%})"
    )

    sc_weights = model.to_sc_weights()
    for i, w in enumerate(sc_weights):
        print(f"  SC weight {i}: shape={tuple(w.shape)}, range=[{w.min():.3f}, {w.max():.3f}]")


if __name__ == "__main__":
    main()
