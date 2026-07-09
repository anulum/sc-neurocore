#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Vertex AI SHD config generator
"""Generate Vertex AI configs for Tim/CNRS SHD follow-up experiments."""

from __future__ import annotations

from pathlib import Path


OUT_DIR = Path("docs/internal/vertex_job_configs")
TRAINING_URI = "gs://gotm-director-ai-training/sc-neurocore-shd-training/"
RESULTS_URI = "gs://gotm-director-ai-training/sc-neurocore-shd-results/"
IMAGE = "europe-west4-docker.pkg.dev/gotm-sc-neurocore/sc-neurocore-training/shd-retrain:latest"


def _env_block(env: dict[str, str]) -> str:
    return "\n".join(
        f'        - name: {name}\n          value: "{value}"' for name, value in env.items()
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
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — internal Vertex AI custom-job config
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

    for seed in range(5):
        out_subdir = f"dcls_max_standard_lif_iter_eps_l1e5_seed{seed}_20260513"
        display_name = f"scn-tim-stdlif-iter-eps-l1e5-seed{seed}-20260513"
        path = OUT_DIR / (f"director_shd_standard_lif_iter_eps_l1e5_seed{seed}_2026_05_13.yaml")
        path.write_text(
            _config(
                display_name,
                out_subdir,
                {
                    "SHD_SEED": str(seed),
                    "SHD_NEURON_MODULE": "standard_lif",
                    "SHD_L1_WEIGHT": "0.00001",
                    "SHD_PRUNE_METHOD": "epsilon",
                    "SHD_PRUNE_PROTOCOL": "iterative_finetune",
                    "SHD_PRUNE_SPARSITY": "0.30",
                    "SHD_PRUNE_EPSILONS": "0.0075,0.01,0.0125,0.015,0.02,0.03,0.04,0.05",
                    "SHD_PRUNE_STEP_FINETUNE_EPOCHS": "15",
                    "SHD_PRUNE_MAX_DEPLOYABLE_DROP": "1.0",
                    "SHD_FINETUNE_EPOCHS": "15",
                },
            ),
            encoding="utf-8",
        )
        written.append(path)

    for seed in range(5):
        out_subdir = f"dcls_max_standard_lif_iter_eps_l1e4_h128_seed{seed}_20260520"
        display_name = f"scn-tim-stdlif-iter-eps-l1e4-h128-seed{seed}-20260520"
        path = OUT_DIR / (
            f"director_shd_standard_lif_iter_eps_l1e4_h128_seed{seed}_2026_05_20.yaml"
        )
        path.write_text(
            _config(
                display_name,
                out_subdir,
                {
                    "SHD_SEED": str(seed),
                    "SHD_NEURON_MODULE": "standard_lif",
                    "SHD_HIDDEN_LAYERS": "128,128",
                    "SHD_L1_WEIGHT": "0.0001",
                    "SHD_PRUNE_METHOD": "epsilon",
                    "SHD_PRUNE_PROTOCOL": "iterative_finetune",
                    "SHD_PRUNE_SPARSITY": "0.30",
                    "SHD_PRUNE_EPSILONS": "0.0075,0.01,0.0125,0.015,0.02,0.03,0.04,0.05",
                    "SHD_PRUNE_STEP_FINETUNE_EPOCHS": "15",
                    "SHD_PRUNE_MAX_DEPLOYABLE_DROP": "1.0",
                    "SHD_FINETUNE_EPOCHS": "15",
                },
            ),
            encoding="utf-8",
        )
        written.append(path)

    for seed in range(5):
        out_subdir = f"dcls_max_standard_lif_iter_eps_l3e5_h256_seed{seed}_20260520"
        display_name = f"scn-tim-stdlif-iter-eps-l3e5-h256-seed{seed}-20260520"
        path = OUT_DIR / (
            f"director_shd_standard_lif_iter_eps_l3e5_h256_seed{seed}_2026_05_20.yaml"
        )
        path.write_text(
            _config(
                display_name,
                out_subdir,
                {
                    "SHD_SEED": str(seed),
                    "SHD_NEURON_MODULE": "standard_lif",
                    "SHD_HIDDEN_LAYERS": "256,256",
                    "SHD_L1_WEIGHT": "0.00003",
                    "SHD_PRUNE_METHOD": "epsilon",
                    "SHD_PRUNE_PROTOCOL": "iterative_finetune",
                    "SHD_PRUNE_SPARSITY": "0.30",
                    "SHD_PRUNE_EPSILONS": "0.0075,0.01,0.0125,0.015,0.02,0.03,0.04,0.05",
                    "SHD_PRUNE_STEP_FINETUNE_EPOCHS": "15",
                    "SHD_PRUNE_MAX_DEPLOYABLE_DROP": "1.0",
                    "SHD_FINETUNE_EPOCHS": "15",
                },
            ),
            encoding="utf-8",
        )
        written.append(path)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
