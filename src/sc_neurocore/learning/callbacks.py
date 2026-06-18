# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Training callbacks for logging metrics to TensorBoard or W&B

"""Training callbacks for logging metrics to TensorBoard or W&B.

Usage::

    from sc_neurocore.learning.callbacks import TensorBoardCallback

    cb = TensorBoardCallback(log_dir="runs/exp1")
    for epoch in range(100):
        cb.log({"loss": 0.5, "accuracy": 0.9}, step=epoch)
    cb.close()
"""

from __future__ import annotations

from typing import Any


class TrainingCallback:
    """Base class for training callbacks."""

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Record a mapping of metric names to values at the given step."""

    def close(self) -> None:
        """Flush and release any resources held by the callback."""


class TensorBoardCallback(TrainingCallback):
    """Log scalars to TensorBoard via ``torch.utils.tensorboard``."""

    def __init__(self, log_dir: str = "runs"):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            from sc_neurocore.exceptions import SCDependencyError

            raise SCDependencyError("TensorBoard requires torch: pip install sc-neurocore[gpu]")
        self._writer = SummaryWriter(log_dir=log_dir)  # type: ignore[no-untyped-call]

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Write each metric as a TensorBoard scalar at the given step."""
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, step)  # type: ignore[no-untyped-call]

    def close(self) -> None:
        """Close the underlying TensorBoard summary writer."""
        self._writer.close()  # type: ignore[no-untyped-call]


class WandBCallback(TrainingCallback):
    """Log metrics to Weights & Biases."""

    def __init__(self, project: str = "sc-neurocore", **init_kwargs: Any):
        try:
            import wandb

            self._wandb = wandb
        except ImportError:
            from sc_neurocore.exceptions import SCDependencyError

            raise SCDependencyError("W&B requires wandb: pip install wandb")
        self._wandb.init(project=project, **init_kwargs)

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Forward the metrics to the active Weights & Biases run."""
        self._wandb.log(metrics, step=step)

    def close(self) -> None:
        """Finish the active Weights & Biases run."""
        self._wandb.finish()


class CSVCallback(TrainingCallback):
    """Log metrics to a CSV file (no dependencies)."""

    def __init__(self, path: str = "metrics.csv"):
        self._path = path
        self._rows: list[dict[str, float | int]] = []

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Buffer one row of metrics for the given step in memory."""
        self._rows.append({"step": step, **metrics})

    def close(self) -> None:
        """Write all buffered metric rows to the CSV file."""
        if not self._rows:
            return
        keys = list(self._rows[0].keys())
        with open(self._path, "w", newline="") as f:
            f.write(",".join(keys) + "\n")
            for row in self._rows:
                f.write(",".join(str(row[k]) for k in keys) + "\n")
