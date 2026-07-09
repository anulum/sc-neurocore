#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# SC-NeuroCore -- ANN-to-SNN conversion demo
#
# Usage:
#   pip install sc-neurocore torch
#   PYTHONPATH=src python examples/ann_to_snn_demo.py

"""Convert a trained PyTorch ANN to a rate-coded SNN and classify."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from sc_neurocore.conversion import convert


def main():
    print("ANN-to-SNN Conversion Demo")
    print("=" * 50)

    # 1. Build and "train" a simple ANN
    torch.manual_seed(42)
    ann = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )
    print(f"\n1. ANN: {ann}")

    # 2. Convert to SNN
    cal_data = torch.randn(50, 8)
    snn = convert(ann, calibration_data=cal_data, T=32)
    print(f"\n2. Converted SNN: {snn.n_layers} layers, T={snn.T}")
    print(f"   Thresholds: {snn.thresholds}")

    # 3. Run inference
    x = np.random.rand(8)
    counts = snn.run(x)
    pred = snn.classify(x)
    print(f"\n3. Input: {x.round(2)}")
    print(f"   Spike counts: {counts}")
    print(f"   Prediction: class {pred}")

    # 4. Batch inference
    x_batch = np.random.rand(20, 8)
    preds = snn.classify(x_batch)
    print(f"\n4. Batch predictions (20 samples): {preds}")

    print("\n" + "=" * 50)
    print("ANN-to-SNN conversion complete.")


if __name__ == "__main__":
    main()
