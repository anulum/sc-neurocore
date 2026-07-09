#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cosim: Q8.8 fixed-point reference vs PyTorch float reference
"""Compare the bit-true Q8.8 SHD reference (tools/shd_q88_reference.py)
against the original PyTorch model on the SHD test set.

This validates the entire fixed-point pipeline before we commit to writing
Verilog modules - if Q8.8 + softplus LUT + sparse int8 weights match PyTorch
within an acceptable accuracy gap (target: <2% absolute), the Verilog
implementation has a high-quality reference for cosim.

Output: classification accuracy comparison + per-sample agreement table.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
TRAINING_REPO = REPO / "data/masquelier_shd/neuromorphic_training-main"
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(TRAINING_REPO))

from sc_neurocore.security.checkpoint_loading import safe_load_legacy_checkpoint

os.environ["WANDB_MODE"] = "disabled"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _require_valid_checkpoint_sha256(digest: str) -> str:
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError("checkpoint_sha256 must be exactly 64 hexadecimal characters")
    return digest


def _require_legacy_checkpoint_contract(ckpt: Any) -> dict[str, Any]:
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint must be a dictionary payload")

    net = ckpt.get("net")
    if not isinstance(net, dict) or not net:
        raise ValueError("checkpoint['net'] must be a non-empty dictionary")

    torch = importlib.import_module("torch")
    for key, value in net.items():
        if not isinstance(key, str) or not key:
            raise ValueError("checkpoint['net'] keys must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint['net'][{key!r}] must be a torch.Tensor")

    raw_acc = ckpt.get("acc")
    if raw_acc is None:
        raise ValueError("checkpoint must include 'acc' metadata")
    try:
        acc = float(raw_acc)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint['acc'] must be numeric") from exc
    if not np.isfinite(acc):
        raise ValueError("checkpoint['acc'] must be finite")
    return ckpt


def get_pytorch_predictions(
    model: Any, test_loader: Any, device: Any, stride: int
) -> list[dict[str, Any]]:
    """Run PyTorch model on stride-sampled test set.

    Returns every `stride`-th sample (e.g. stride=75 gives ~30 samples spanning
    all 20 classes for SHD test set of 2264 samples). This avoids the
    sorted-by-class bias of "first N samples".

    MUST mirror trainer.test() exactly: model.eval() + round_pos() at start,
    reset_states (not functional.reset_net), identical input shape handling.
    """
    torch = importlib.import_module("torch")
    reset_states = importlib.import_module("src.utils").reset_states

    model.eval()
    if hasattr(model, "round_pos"):
        model.round_pos()  # CRITICAL: same as trainer.test() line 70-71

    results = []
    sample_idx = 0
    with torch.no_grad():
        for x, label, *_ in test_loader:
            # trainer.test() does: inputs.permute(1,0,2).float().to(device)
            x = x.permute(1, 0, 2).float().to(device)
            label = label.to(device)
            reset_states(model=model)
            out = model(x)
            logits = out.sum(0)
            preds = logits.argmax(1)
            for b in range(x.shape[1]):
                if sample_idx % stride == 0:
                    spikes = x[:, b, :].cpu().numpy().astype(np.int8)
                    results.append(
                        {
                            "sample_idx": sample_idx,
                            "label": int(label[b].item()),
                            "pytorch_pred": int(preds[b].item()),
                            "input_spikes": spikes,
                        }
                    )
                sample_idx += 1
    return results


def main(checkpoint_path: str, artifacts_dir: str, stride: int, *, checkpoint_sha256: str) -> None:
    torch = importlib.import_module("torch")
    Config = importlib.import_module("configs.config_SHD").Config
    shd_q88_reference = importlib.import_module("shd_q88_reference")
    load_artifacts = shd_q88_reference.load_artifacts
    run_inference_q88 = shd_q88_reference.run_inference_q88
    load_dataset = importlib.import_module("src.datasets").load_dataset
    SNN_axonal_feedforward_delays = importlib.import_module(
        "src.SHD.snn"
    ).SNN_axonal_feedforward_delays
    trainer_test = importlib.import_module("src.SHD.trainer").test

    try:
        import wandb
    except ModuleNotFoundError:
        wandb = None
    if wandb is not None:
        wandb.init(mode="disabled")

    print(f"Loading checkpoint: {checkpoint_path}")
    config = Config()
    config.datasets_path = str((TRAINING_REPO / config.datasets_path).resolve())
    config.hidden_layers = [128, 128]
    config.DCLSversion = "max"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SNN_axonal_feedforward_delays(config).to(device)
    trusted_digest = _require_valid_checkpoint_sha256(checkpoint_sha256)
    ckpt = safe_load_legacy_checkpoint(
        checkpoint_path,
        trusted_sha256={Path(checkpoint_path).name: trusted_digest},
        map_location=device,
    )
    ckpt = _require_legacy_checkpoint_contract(ckpt)
    model.load_state_dict(ckpt["net"])
    print(f"  PyTorch val_acc={ckpt['acc']:.2f}%, sigma={ckpt.get('sigma', 'n/a')}")

    # Set sigma to native value from checkpoint (don't force change).
    # Do NOT round_pos() - at sigma=0.23 the trained delays already snap
    # to integers via the narrow kernel. The cloud eval that gave 75.2%
    # used `set_sigma(model, 0.23); test(...)` WITHOUT round_pos.
    dcls_module = importlib.import_module("src.modules").dcls_module

    for m in model.modules():
        if isinstance(m, dcls_module) and hasattr(m, "SIG"):
            m.SIG.data.fill_(float(ckpt.get("sigma", 0.23)))
            m.SIG.requires_grad = False

    # Load test set
    _, _, test_loader = load_dataset(config)
    print(f"Test samples available: {len(test_loader.dataset)}")

    # First: full test accuracy via the canonical trainer.test() - this MUST
    # match the cloud-reported number (75.21% for dcls_max last.pth at sigma=0.23).
    print("\n[1/3] Sanity check via trainer.test() (full test set)...")
    t0 = time.perf_counter()
    test_acc, test_loss = trainer_test(test_loader, model, 0, device, config)
    print(
        f"  trainer.test() returned acc={test_acc:.2f}% loss={test_loss:.4f} "
        f"in {time.perf_counter() - t0:.1f}s"
    )
    print(
        f"  Cloud-reported number: 75.21% - {'MATCH' if abs(test_acc - 75.21) < 1.0 else 'MISMATCH'}"
    )

    # Now run per-sample (every `stride`th) to capture inputs for Q8.8 cosim
    print(f"\n[2/3] Running per-sample PyTorch inference (every {stride}th sample)...")
    t0 = time.perf_counter()
    pytorch_results = get_pytorch_predictions(model, test_loader, device, stride)
    pt_time = time.perf_counter() - t0
    print(f"  done in {pt_time:.1f}s, {len(pytorch_results)} samples kept")
    label_dist: dict[int, int] = {}
    for r in pytorch_results:
        label_dist[r["label"]] = label_dist.get(r["label"], 0) + 1
    print(f"  label distribution: {dict(sorted(label_dist.items()))}")
    pt_acc = sum(r["label"] == r["pytorch_pred"] for r in pytorch_results) / len(pytorch_results)
    print(f"  PyTorch per-sample accuracy on stratified subset: {pt_acc * 100:.2f}%")

    # Q8.8 reference
    print("\n[3/3] Running Q8.8 reference...")
    net = load_artifacts(artifacts_dir)
    t0 = time.perf_counter()
    q88_preds = []
    for i, r in enumerate(pytorch_results):
        pred = run_inference_q88(net, r["input_spikes"])
        q88_preds.append(pred)
        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  {i + 1}/{len(pytorch_results)} done, {elapsed:.0f}s elapsed "
                f"({elapsed / (i + 1) * 1000:.0f} ms/sample)"
            )
    q88_time = time.perf_counter() - t0
    q88_acc = sum(pytorch_results[i]["label"] == q88_preds[i] for i in range(len(q88_preds))) / len(
        q88_preds
    )
    print(f"  done in {q88_time:.1f}s ({q88_time / len(q88_preds) * 1000:.0f} ms/sample)")
    print(f"  Q8.8 test accuracy on first {len(q88_preds)} samples: {q88_acc * 100:.2f}%")

    # Per-sample agreement
    matches = sum(pytorch_results[i]["pytorch_pred"] == q88_preds[i] for i in range(len(q88_preds)))
    agree = matches / len(q88_preds)
    print("\n=== Cosim Result ===")
    print(f"PyTorch accuracy:  {pt_acc * 100:.2f}%")
    print(f"Q8.8 accuracy:     {q88_acc * 100:.2f}%")
    print(f"Accuracy gap:      {(pt_acc - q88_acc) * 100:+.2f}%")
    print(
        f"Per-sample agreement (PyTorch == Q8.8): {agree * 100:.2f}% ({matches}/{len(q88_preds)})"
    )

    # Disagreements
    disagreements = []
    for i, r in enumerate(pytorch_results):
        if r["pytorch_pred"] != q88_preds[i]:
            disagreements.append((i, r["label"], r["pytorch_pred"], q88_preds[i]))
    if disagreements:
        print("\nDisagreements (first 10):")
        print(f"  {'sample':>6} {'label':>5} {'pytorch':>7} {'q88':>4}")
        for sample, label, pt, q in disagreements[:10]:
            mark_pt = "✓" if pt == label else "✗"
            mark_q = "✓" if q == label else "✗"
            print(f"  {sample:>6} {label:>5} {pt:>7}{mark_pt} {q:>4}{mark_q}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=f"{REPO}/data/masquelier_shd/cloud_results/dcls_max/dcls_max/last.pth",
    )
    parser.add_argument(
        "--artifacts", default=f"{REPO}/data/masquelier_shd/fpga_artifacts/dcls_max/"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=75,
        help="Stride between sampled test items (75 -> ~30 samples spanning all classes)",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        help="Expected SHA-256 digest for the legacy SHD metadata checkpoint",
    )
    args = parser.parse_args()
    main(
        args.checkpoint,
        args.artifacts,
        args.stride,
        checkpoint_sha256=args.checkpoint_sha256,
    )
