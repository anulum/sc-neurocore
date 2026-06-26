# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for training callbacks

"""Tests for training callbacks."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from sc_neurocore.learning.callbacks import CSVCallback, TrainingCallback


def test_base_callback_is_noop() -> None:
    cb = TrainingCallback()
    cb.log({"loss": 1.0}, step=0)
    cb.close()


def test_csv_callback_writes_header_and_rows(tmp_path: Path) -> None:
    path = str(tmp_path / "metrics.csv")
    cb = CSVCallback(path=path)
    cb.log({"loss": 0.5, "acc": 0.9}, step=0)
    cb.log({"loss": 0.3, "acc": 0.95}, step=1)
    cb.close()

    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["step", "loss", "acc"]
        row1 = next(reader)
        assert row1[0] == "0"
        row2 = next(reader)
        assert row2[0] == "1"


def test_csv_callback_close_empty_is_noop(tmp_path: Path) -> None:
    path = str(tmp_path / "empty.csv")
    cb = CSVCallback(path=path)
    cb.close()
    assert not (tmp_path / "empty.csv").exists()


def test_tensorboard_callback_raises_without_torch() -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401

        pytest.skip("torch is installed")
    except ImportError:
        pass

    from sc_neurocore.exceptions import SCDependencyError

    with pytest.raises(SCDependencyError):
        from sc_neurocore.learning.callbacks import TensorBoardCallback

        TensorBoardCallback()


def test_wandb_callback_raises_without_wandb() -> None:
    try:
        import wandb  # noqa: F401

        pytest.skip("wandb is installed")
    except ImportError:
        pass

    from sc_neurocore.exceptions import SCDependencyError

    with pytest.raises(SCDependencyError):
        from sc_neurocore.learning.callbacks import WandBCallback

        WandBCallback()


def test_tensorboard_callback_with_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_writer: Any = MagicMock()
    summary_writer: Any = MagicMock(return_value=mock_writer)
    mock_tb = ModuleType("torch.utils.tensorboard")
    mock_tb_any: Any = mock_tb
    mock_tb_any.SummaryWriter = summary_writer

    mock_torch = ModuleType("torch")
    mock_utils = ModuleType("torch.utils")
    mock_utils_any: Any = mock_utils
    mock_utils_any.tensorboard = mock_tb
    mock_torch_any: Any = mock_torch
    mock_torch_any.utils = mock_utils

    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", mock_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", mock_tb)

    from sc_neurocore.learning.callbacks import TensorBoardCallback

    cb = TensorBoardCallback(log_dir=str(tmp_path))
    cb.log({"loss": 0.5}, step=0)
    mock_writer.add_scalar.assert_called_once_with("loss", 0.5, 0)
    cb.close()
    mock_writer.close.assert_called_once()


def test_wandb_callback_with_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_wandb: Any = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", mock_wandb)

    from sc_neurocore.learning.callbacks import WandBCallback

    cb = WandBCallback(project="test-proj")
    mock_wandb.init.assert_called_once_with(project="test-proj")
    cb.log({"acc": 0.9}, step=1)
    mock_wandb.log.assert_called_once_with({"acc": 0.9}, step=1)
    cb.close()
    mock_wandb.finish.assert_called_once()
