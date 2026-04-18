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
Verilog modules — if Q8.8 + softplus LUT + sparse int8 weights match PyTorch
within an acceptable accuracy gap (target: <2% absolute), the Verilog
implementation has a high-quality reference for cosim.

Output: classification accuracy comparison + per-sample agreement table.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

REPO = "/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SC-NEUROCORE"
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "data/masquelier_shd/neuromorphic_training-main"))
os.chdir(os.path.join(REPO, "data/masquelier_shd/neuromorphic_training-main"))

from shd_q88_reference import load_artifacts, run_inference_q88  # noqa: E402

import torch  # noqa: E402

os.environ["WANDB_MODE"] = "disabled"

import wandb  # noqa: E402

wandb.init(mode="disabled")

from configs.config_SHD import Config  # noqa: E402
from src.datasets import load_dataset  # noqa: E402
from src.SHD.snn import SNN_axonal_feedforward_delays  # noqa: E402
from src.SHD.trainer import test as trainer_test  # noqa: E402
from src.utils import reset_states  # noqa: E402


def get_pytorch_predictions(model, test_loader, device, stride: int):
    """Run PyTorch model on stride-sampled test set.

    Returns every `stride`-th sample (e.g. stride=75 gives ~30 samples spanning
    all 20 classes for SHD test set of 2264 samples). This avoids the
    sorted-by-class bias of "first N samples".

    MUST mirror trainer.test() exactly: model.eval() + round_pos() at start,
    reset_states (not functional.reset_net), identical input shape handling.
    """
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


def main(checkpoint_path: str, artifacts_dir: str, stride: int):
    print(f"Loading checkpoint: {checkpoint_path}")
    config = Config()
    config.hidden_layers = [128, 128]
    config.DCLSversion = "max"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SNN_axonal_feedforward_delays(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["net"])
    print(f"  PyTorch val_acc={ckpt['acc']:.2f}%, sigma={ckpt.get('sigma', 'n/a')}")

    # Set sigma to native value from checkpoint (don't force change).
    # Do NOT round_pos() — at sigma=0.23 the trained delays already snap
    # to integers via the narrow kernel. The cloud eval that gave 75.2%
    # used `set_sigma(model, 0.23); test(...)` WITHOUT round_pos.
    from src.modules import dcls_module

    for m in model.modules():
        if isinstance(m, dcls_module) and hasattr(m, "SIG"):
            m.SIG.data.fill_(float(ckpt.get("sigma", 0.23)))
            m.SIG.requires_grad = False

    # Load test set
    _, _, test_loader = load_dataset(config)
    print(f"Test samples available: {len(test_loader.dataset)}")

    # First: full test accuracy via the canonical trainer.test() — this MUST
    # match the cloud-reported number (75.21% for dcls_max last.pth at sigma=0.23).
    print("\n[1/3] Sanity check via trainer.test() (full test set)...")
    t0 = time.perf_counter()
    test_acc, test_loss = trainer_test(test_loader, model, 0, device, config)
    print(
        f"  trainer.test() returned acc={test_acc:.2f}% loss={test_loss:.4f} "
        f"in {time.perf_counter() - t0:.1f}s"
    )
    print(
        f"  Cloud-reported number: 75.21% — {'MATCH' if abs(test_acc - 75.21) < 1.0 else 'MISMATCH'}"
    )

    # Now run per-sample (every `stride`th) to capture inputs for Q8.8 cosim
    print(f"\n[2/3] Running per-sample PyTorch inference (every {stride}th sample)...")
    t0 = time.perf_counter()
    pytorch_results = get_pytorch_predictions(model, test_loader, device, stride)
    pt_time = time.perf_counter() - t0
    print(f"  done in {pt_time:.1f}s, {len(pytorch_results)} samples kept")
    label_dist = {}
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
    args = parser.parse_args()
    main(args.checkpoint, args.artifacts, args.stride)
