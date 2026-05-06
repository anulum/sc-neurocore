#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 ANULUM
#
# This file is part of sc-neurocore.
# See the LICENSE file in the project root for full license text.
# Commercial licensing is available; contact protoscience@anulum.li.
"""Generate Vertex AI configs for Tim/CNRS SHD follow-up experiments."""

from __future__ import annotations

from pathlib import Path


OUT_DIR = Path("docs/internal/vertex_job_configs")
TRAINING_URI = "gs://gotm-director-ai-training/sc-neurocore-shd-training/"
RESULTS_URI = "gs://gotm-director-ai-training/sc-neurocore-shd-results/"
IMAGE = (
    "europe-west4-docker.pkg.dev/gotm-sc-neurocore/"
    "sc-neurocore-training/shd-retrain:latest"
)


def _env_block(env: dict[str, str]) -> str:
    return "\n".join(
        f"        - name: {name}\n          value: \"{value}\""
        for name, value in env.items()
    )


def _config(display_name: str, out_subdir: str, env: dict[str, str]) -> str:
    env = {
        "SHD_SIGMA_INIT": "15.0",
        "SHD_SIGMA_FINAL": "0.0",
        "SHD_ROUND_EACH_EPOCH": "0",
        "SHD_LAMBDA_DELAY": "0.0",
        "SHD_FINETUNE_EPOCHS": "0",
        "SHD_EPOCHS": "150",
        "SHD_OUTPUT_SUBDIR": out_subdir,
        "PYTHONUNBUFFERED": "1",
        **env,
    }
    return f"""# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 ANULUM
#
# Internal Vertex AI custom-job config for SC-NeuroCore SHD follow-up.
# Display name: {display_name}
workerPoolSpecs:
  - machineSpec:
      machineType: n1-standard-8
      acceleratorType: NVIDIA_TESLA_T4
      acceleratorCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 100
    replicaCount: 1
    containerSpec:
      imageUri: {IMAGE}
      command:
        - /bin/bash
        - -c
      args:
        - |
          set -e
          mkdir -p /workspace/training
          echo "=== Downloading SC-NeuroCore SHD training payload ==="
          gsutil -m rsync -r {TRAINING_URI} /workspace/training/
          cd /workspace/training
          mkdir -p Datasets/SHD/download
          echo "=== Starting {display_name} ==="
          python3 -u train_dcls_max.py 2>&1
          echo "=== Uploading SC-NeuroCore result artifacts ==="
          OUTDIR="exp/SHD/SNN_axonal_feedforward_delays/{out_subdir}"
          if [ -d "$OUTDIR" ]; then
            gsutil -m cp -r "$OUTDIR"/* {RESULTS_URI}{out_subdir}/
          else
            echo "No expected output directory: $OUTDIR"
            find exp -maxdepth 5 -type d -name "dcls_max*" 2>/dev/null || true
            exit 1
          fi
          echo "=== DONE ==="
      env:
{_env_block(env)}
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for seed in range(5):
        out_subdir = f"dcls_max_standard_lif_seed{seed}"
        display_name = f"scn-tim-standard-lif-seed{seed}-20260506"
        path = OUT_DIR / f"director_shd_standard_lif_seed{seed}_2026_05_06.yaml"
        path.write_text(
            _config(
                display_name,
                out_subdir,
                {
                    "SHD_SEED": str(seed),
                    "SHD_NEURON_MODULE": "standard_lif",
                    "SHD_L1_WEIGHT": "0.0",
                },
            ),
            encoding="utf-8",
        )
        written.append(path)

    for seed in range(5):
        out_subdir = f"dcls_max_epsilon_prune_l1e5_seed{seed}"
        display_name = f"scn-tim-eps-prune-l1e5-seed{seed}-20260506"
        path = OUT_DIR / f"director_shd_epsilon_prune_l1e5_seed{seed}_2026_05_06.yaml"
        path.write_text(
            _config(
                display_name,
                out_subdir,
                {
                    "SHD_SEED": str(seed),
                    "SHD_NEURON_MODULE": "vmin_lif",
                    "SHD_L1_WEIGHT": "0.00001",
                    "SHD_PRUNE_METHOD": "epsilon",
                    "SHD_PRUNE_SPARSITY": "0.9",
                    "SHD_PRUNE_EPSILON": "0.01",
                    "SHD_PRUNE_EPSILON_GROWTH": "1.25",
                    "SHD_FINETUNE_EPOCHS": "20",
                },
            ),
            encoding="utf-8",
        )
        written.append(path)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
