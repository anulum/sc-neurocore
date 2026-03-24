# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NeuroBench benchmark task definitions

"""Built-in benchmark task definitions aligned with NeuroBench.

Each task defines: dataset, input shape, number of classes/outputs,
evaluation metric, and baseline performance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkTask:
    """Definition of a benchmark task."""

    name: str
    description: str
    input_shape: tuple[int, ...]
    n_classes: int
    metric: str
    neurobench_id: str
    dataset: str
    baseline_accuracy: float


TASKS = {
    "keyword_spotting": BenchmarkTask(
        name="Keyword Spotting",
        description="12-class spoken keyword classification (Google Speech Commands v2)",
        input_shape=(16000,),
        n_classes=12,
        metric="accuracy",
        neurobench_id="keyword_spotting",
        dataset="speech_commands_v2",
        baseline_accuracy=0.92,
    ),
    "dvs_gesture": BenchmarkTask(
        name="DVS Gesture Recognition",
        description="11-class gesture classification from DVS128 event camera",
        input_shape=(128, 128),
        n_classes=11,
        metric="accuracy",
        neurobench_id="dvs_gesture",
        dataset="dvs_gesture",
        baseline_accuracy=0.95,
    ),
    "heartbeat_anomaly": BenchmarkTask(
        name="Heartbeat Anomaly Detection",
        description="Binary anomaly detection on MIT-BIH ECG dataset",
        input_shape=(187,),
        n_classes=2,
        metric="accuracy",
        neurobench_id="ecg_anomaly",
        dataset="mit_bih",
        baseline_accuracy=0.97,
    ),
    "mnist": BenchmarkTask(
        name="MNIST Classification",
        description="10-class handwritten digit classification",
        input_shape=(784,),
        n_classes=10,
        metric="accuracy",
        neurobench_id="mnist",
        dataset="mnist",
        baseline_accuracy=0.99,
    ),
    "shd": BenchmarkTask(
        name="Spiking Heidelberg Digits",
        description="20-class spoken digit classification (spiking audio)",
        input_shape=(700,),
        n_classes=20,
        metric="accuracy",
        neurobench_id="shd",
        dataset="shd",
        baseline_accuracy=0.85,
    ),
}
