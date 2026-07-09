#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# SC-NeuroCore — Validate bipolar SC readout on a real MNIST checkpoint

"""Validate a real MNIST checkpoint against a bipolar SC readout bridge.

This tool uses ``torchvision.datasets.MNIST`` by default so the dataset path
matches the training examples. The direct IDX loader remains available for
low-level file-contract tests and explicit fallback validation. It measures:

* float checkpoint accuracy for the full ``ConvSpikingNet``;
* bipolar SC final-readout accuracy using ``VectorizedSCLayer`` loaded from
  ``to_sc_weights(encoding="bipolar")``;
* prediction agreement between the full float checkpoint and the SC readout.

The SC path validates the final trained classifier readout on real checkpoint
features and real dataset labels. It is not a full convolutional bitstream
replacement for every preceding layer.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer
from sc_neurocore.security.checkpoint_loading import safe_load_checkpoint
from sc_neurocore.training.snn_modules import ConvSpikingNet

MNIST_IMAGE_MAGIC = 2051
MNIST_LABEL_MAGIC = 2049
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class ValidationResult:
    checkpoint: str
    data_dir: str
    samples: int
    timesteps: int
    bitstream_length: int
    seed: int
    dataset_loader: str
    float_checkpoint_accuracy: float
    sc_final_readout_accuracy: float
    float_sc_prediction_agreement: float
    min_sc_accuracy: float
    min_agreement: float
    passed: bool
    scope: str


def _read_idx_images(path: Path, limit: int) -> np.ndarray[Any, np.dtype[np.float32]]:
    raw = path.read_bytes()
    if len(raw) < 16:
        raise ValueError(f"MNIST image file is truncated: {path}")
    magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != MNIST_IMAGE_MAGIC:
        raise ValueError(f"invalid MNIST image magic {magic} in {path}")
    n_items = min(limit, count)
    expected = 16 + count * rows * cols
    if len(raw) < expected:
        raise ValueError(f"MNIST image file has incomplete payload: {path}")
    images = np.frombuffer(raw, dtype=np.uint8, offset=16, count=n_items * rows * cols)
    return (images.reshape(n_items, rows, cols).astype(np.float32) / 255.0 - 0.1307) / 0.3081


def _read_idx_labels(path: Path, limit: int) -> np.ndarray[Any, np.dtype[np.int64]]:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"MNIST label file is truncated: {path}")
    magic, count = struct.unpack(">II", raw[:8])
    if magic != MNIST_LABEL_MAGIC:
        raise ValueError(f"invalid MNIST label magic {magic} in {path}")
    n_items = min(limit, count)
    expected = 8 + count
    if len(raw) < expected:
        raise ValueError(f"MNIST label file has incomplete payload: {path}")
    return np.frombuffer(raw, dtype=np.uint8, offset=8, count=n_items).astype(np.int64)


def load_mnist_idx(
    data_dir: Path, samples: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Load normalized MNIST test images and labels from local IDX files."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    raw_dir = data_dir / "MNIST" / "raw"
    images = _read_idx_images(raw_dir / "t10k-images-idx3-ubyte", samples)
    labels = _read_idx_labels(raw_dir / "t10k-labels-idx1-ubyte", samples)
    n_items = min(images.shape[0], labels.shape[0])
    if n_items < samples:
        raise ValueError(f"requested {samples} samples but only {n_items} are available")
    return images[:samples], labels[:samples]


def load_mnist_torchvision(
    data_dir: Path,
    samples: int,
    *,
    download: bool = False,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Load normalized MNIST test images through torchvision."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover - dependency is verified in CI/session
        raise RuntimeError("torchvision is required for the default MNIST validator path") from exc

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = datasets.MNIST(str(data_dir), train=False, download=download, transform=transform)
    if len(dataset) < samples:
        raise ValueError(f"requested {samples} samples but only {len(dataset)} are available")
    images: list[np.ndarray[Any, Any]] = []
    labels: list[int] = []
    for idx in range(samples):
        image, label = dataset[idx]
        images.append(image.squeeze(0).numpy().astype(np.float32))
        labels.append(int(label))
    return np.stack(images, axis=0), np.asarray(labels, dtype=np.int64)


def _state_dict_from_checkpoint(
    path: Path,
    *,
    trusted_sha256: dict[str, str],
) -> dict[str, torch.Tensor]:
    checkpoint = safe_load_checkpoint(path, trusted_sha256=trusted_sha256, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    if (
        not isinstance(state, dict)
        or not state
        or not all(isinstance(k, str) and k for k in state)
        or not all(isinstance(v, torch.Tensor) for v in state.values())
    ):
        raise ValueError(f"checkpoint does not contain a state_dict: {path}")
    for key, value in state.items():
        if value.numel() == 0:
            raise ValueError(f"checkpoint tensor '{key}' must be non-empty")
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise ValueError(f"checkpoint tensor '{key}' contains non-finite values")
    return {key: value for key, value in state.items()}


def load_conv_checkpoint(
    path: Path,
    *,
    trusted_sha256: dict[str, str],
) -> ConvSpikingNet:
    """Load ``ConvSpikingNet`` while preserving learned dynamics flags."""
    state = _state_dict_from_checkpoint(path, trusted_sha256=trusted_sha256)
    learn_beta = any(key.endswith("._beta_logit") for key in state)
    learn_threshold = any(key.endswith("._threshold_log") for key in state)
    model = ConvSpikingNet(
        n_output=10,
        learn_beta=learn_beta,
        learn_threshold=learn_threshold,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def _lif3_rates(model: ConvSpikingNet, x: torch.Tensor, timesteps: int) -> np.ndarray[Any, Any]:
    v1 = torch.zeros(1, 32, 24, 24)
    v2 = torch.zeros(1, 64, 8, 8)
    v3 = torch.zeros(1, 128)
    spike_count = torch.zeros(128)
    for t in range(timesteps):
        h = model.conv1(x[t])
        spk, v1 = model.lif1(h, v1)
        h = model.pool1(spk)
        h = model.conv2(h)
        spk, v2 = model.lif2(h, v2)
        h = model.pool2(spk)
        h = h.flatten(1)
        h = model.fc1(h)
        spk, v3 = model.lif3(h, v3)
        spike_count += spk.squeeze(0)
    return (spike_count / timesteps).numpy().astype(np.float64)


def validate_checkpoint(
    *,
    checkpoint: Path,
    data_dir: Path,
    samples: int,
    timesteps: int,
    bitstream_length: int,
    seed: int,
    min_sc_accuracy: float,
    min_agreement: float,
    dataset_loader: str = "torchvision",
    download: bool = False,
    checkpoint_sha256: str | None = None,
) -> ValidationResult:
    """Run the real checkpoint/dataset SC final-readout validation."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if bitstream_length <= 0:
        raise ValueError("bitstream_length must be positive")
    if not 0.0 <= min_sc_accuracy <= 1.0:
        raise ValueError("min_sc_accuracy must be in [0, 1]")
    if not 0.0 <= min_agreement <= 1.0:
        raise ValueError("min_agreement must be in [0, 1]")
    if not checkpoint_sha256:
        raise ValueError("checkpoint_sha256 is required for trusted checkpoint loading")
    if not _SHA256_RE.fullmatch(checkpoint_sha256):
        raise ValueError("checkpoint_sha256 must be exactly 64 hexadecimal characters")

    if dataset_loader == "torchvision":
        images, labels = load_mnist_torchvision(data_dir, samples, download=download)
    elif dataset_loader == "idx":
        images, labels = load_mnist_idx(data_dir, samples)
    else:
        raise ValueError("dataset_loader must be 'torchvision' or 'idx'")
    trusted_sha256 = {checkpoint.name: checkpoint_sha256}
    model = load_conv_checkpoint(checkpoint, trusted_sha256=trusted_sha256)
    sc_readout = VectorizedSCLayer.from_exported_weights(
        model.to_sc_weights(encoding="bipolar")[-1],
        length=bitstream_length,
        use_gpu=False,
        seed=seed,
    )

    float_correct = 0
    sc_correct = 0
    agreement = 0
    with torch.inference_mode():
        for image, label in zip(images, labels, strict=True):
            x0 = torch.from_numpy(image).view(1, 1, 28, 28)
            x = x0.unsqueeze(0).expand(timesteps, -1, -1, -1, -1)
            spike_counts, _ = model(x)
            float_prediction = int(spike_counts.argmax(1).item())
            lif3_rates = _lif3_rates(model, x, timesteps)
            sc_input = (2.0 * lif3_rates - 1.0).tolist()
            sc_scores = sc_readout.forward(sc_input)
            sc_prediction = int(np.argmax(sc_scores))
            label_int = int(label)
            float_correct += float_prediction == label_int
            sc_correct += sc_prediction == label_int
            agreement += sc_prediction == float_prediction

    float_accuracy = float_correct / samples
    sc_accuracy = sc_correct / samples
    agreement_rate = agreement / samples
    passed = sc_accuracy >= min_sc_accuracy and agreement_rate >= min_agreement
    return ValidationResult(
        checkpoint=str(checkpoint),
        data_dir=str(data_dir),
        samples=samples,
        timesteps=timesteps,
        bitstream_length=bitstream_length,
        seed=seed,
        dataset_loader=dataset_loader,
        float_checkpoint_accuracy=float_accuracy,
        sc_final_readout_accuracy=sc_accuracy,
        float_sc_prediction_agreement=agreement_rate,
        min_sc_accuracy=min_sc_accuracy,
        min_agreement=min_agreement,
        passed=passed,
        scope=(
            f"real MNIST test data via {dataset_loader}, real ConvSpikingNet checkpoint, "
            "bipolar SC final-readout bridge"
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("conv_spiking_net_best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=25)
    parser.add_argument("--bitstream-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--min-sc-accuracy", type=float, default=0.70)
    parser.add_argument("--min-agreement", type=float, default=0.65)
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        help="Expected SHA-256 digest for the checkpoint before PyTorch deserialisation",
    )
    parser.add_argument("--dataset-loader", choices=("torchvision", "idx"), default="torchvision")
    parser.add_argument(
        "--download", action="store_true", help="Allow torchvision to download MNIST"
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    result = validate_checkpoint(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        samples=args.samples,
        timesteps=args.timesteps,
        bitstream_length=args.bitstream_length,
        seed=args.seed,
        min_sc_accuracy=args.min_sc_accuracy,
        min_agreement=args.min_agreement,
        dataset_loader=args.dataset_loader,
        download=args.download,
        checkpoint_sha256=args.checkpoint_sha256,
    )
    payload = asdict(result)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
