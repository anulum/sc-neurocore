# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Tim/CNRS iterative SHD pruning helpers
"""Regression tests for the Tim/CNRS iterative SHD pruning protocol helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "data/masquelier_shd/train_dcls_max.py"


def _load_training_helpers():
    for mod_name in (
        "torch",
        "wandb",
        "spikingjelly",
        "spikingjelly.activation_based",
        "configs",
        "configs.config_SHD",
        "src",
        "src.datasets",
        "src.modules",
        "src.neurons",
        "src.SHD",
        "src.SHD.snn",
        "src.SHD.trainer",
        "src.utils",
    ):
        sys.modules.setdefault(mod_name, types.ModuleType(mod_name))

    sys.modules["torch"].nn = types.SimpleNamespace(Module=object)
    sys.modules["torch"].Tensor = object
    sys.modules["wandb"].init = lambda **_: None
    sys.modules["spikingjelly.activation_based"].neuron = types.SimpleNamespace(
        LIFNode=object
    )
    sys.modules["spikingjelly.activation_based"].surrogate = types.SimpleNamespace(
        Sigmoid=lambda: object()
    )
    sys.modules["configs.config_SHD"].Config = type("Config", (), {})
    sys.modules["src.datasets"].SHD_dataloaders = lambda _: (None, None, None)
    sys.modules["src.modules"].dcls_module = type("dcls_module", (), {})
    sys.modules["src.neurons"].Vmin_LIFNode = type("Vmin_LIFNode", (), {})
    sys.modules["src.SHD.snn"].SNN_axonal_feedforward_delays = type("Net", (), {})
    sys.modules["src.SHD.trainer"].test = lambda *_, **__: (0.0, 0.0)
    sys.modules["src.SHD.trainer"].init_optim_sche = lambda *_, **__: (None, None)
    sys.modules["src.SHD.trainer"].count_parameters = lambda _: 0
    sys.modules["src.utils"].seed_everything = lambda *_, **__: None

    spec = importlib.util.spec_from_file_location("train_dcls_max_helpers", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    source = SCRIPT.read_text(encoding="utf-8")
    exec(source.split('if __name__ == "__main__":')[0], module.__dict__)
    return module


def test_explicit_iterative_epsilon_schedule_is_strict_and_preserved() -> None:
    helpers = _load_training_helpers()

    schedule = helpers.parse_epsilon_schedule("0.0075,0.01,0.0125,0.02")

    assert helpers.iterative_epsilon_schedule(
        initial_epsilon=0.01,
        target_sparsity=0.30,
        growth=1.25,
        max_steps=20,
        explicit_schedule=schedule,
    ) == [0.0075, 0.01, 0.0125, 0.02]


def test_iterative_epsilon_schedule_rejects_invalid_ladders() -> None:
    helpers = _load_training_helpers()

    with pytest.raises(ValueError, match="strictly increasing"):
        helpers.parse_epsilon_schedule("0.01,0.01")

    with pytest.raises(ValueError, match="positive"):
        helpers.parse_epsilon_schedule("-0.01")

    with pytest.raises(ValueError, match="growth"):
        helpers.iterative_epsilon_schedule(
            initial_epsilon=0.01,
            target_sparsity=0.30,
            growth=1.0,
            max_steps=20,
            explicit_schedule=[],
        )
