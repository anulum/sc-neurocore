# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for training callbacks."""

from __future__ import annotations

import csv

import pytest

from sc_neurocore.learning.callbacks import CSVCallback, TrainingCallback


def test_base_callback_is_noop():
    cb = TrainingCallback()
    cb.log({"loss": 1.0}, step=0)
    cb.close()


def test_csv_callback_writes_header_and_rows(tmp_path):
    path = str(tmp_path / "metrics.csv")
    cb = CSVCallback(path=path)
    cb.log({"loss": 0.5, "acc": 0.9}, step=0)
    cb.log({"loss": 0.3, "acc": 0.95}, step=1)
    cb.close()  # CSV is written on close

    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["step", "loss", "acc"]
        row1 = next(reader)
        assert row1[0] == "0"
        row2 = next(reader)
        assert row2[0] == "1"


def test_tensorboard_callback_raises_without_torch():
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401

        pytest.skip("torch is installed")
    except ImportError:
        pass

    from sc_neurocore.exceptions import SCDependencyError

    with pytest.raises(SCDependencyError):
        from sc_neurocore.learning.callbacks import TensorBoardCallback

        TensorBoardCallback()


def test_wandb_callback_raises_without_wandb():
    try:
        import wandb  # noqa: F401

        pytest.skip("wandb is installed")
    except ImportError:
        pass

    from sc_neurocore.exceptions import SCDependencyError

    with pytest.raises(SCDependencyError):
        from sc_neurocore.learning.callbacks import WandBCallback

        WandBCallback()
