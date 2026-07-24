# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_network_execution_paths.py

from __future__ import annotations

"""Execution-path coverage for the Network simulation orchestrator: the Rust
backend body, the MPI dispatch guards, the engine-detection helpers, and the
pure-Python FIM / plasticity / torch-bridge paths."""

import builtins


from pathlib import Path


import sys


from typing import Any


import numpy as np


import pytest


import sc_neurocore.network.network as network_module


from sc_neurocore.network.monitor import RateMonitor, SpikeMonitor, StateMonitor


from sc_neurocore.network.network import Network


from sc_neurocore.network.population import Population


from sc_neurocore.network.projection import Projection


from sc_neurocore.network.stimulus import StepCurrent

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_MODEL = "AdExNeuron"


class _FakeNetworkRunner:
    """Stand-in Rust runner returning crafted voltages and packed spike events,
    so the Rust result-decode body runs deterministically without the engine."""

    instances: list[_FakeNetworkRunner] = []

    def __init__(self) -> None:
        self.added_models: list[str] = []
        type(self).instances.append(self)

    @staticmethod
    def supported_models() -> set[str]:
        return {_MODEL}

    def add_population(self, model_name: str, n: int) -> int:
        self.added_models.append(model_name)
        return 0

    def add_projection(self, *args: object) -> None:
        return None

    def run(self, n_steps: int) -> dict[str, object]:
        # voltages[0] length must equal the population size to sync back; the
        # packed spike encodes neuron 1 firing at timestep 2.
        return {"voltages": [[0.1, 0.2, 0.3]], "spike_data": [[(1 << 32) | 2]]}


def _install_fake_rust_engine(monkeypatch: pytest.MonkeyPatch) -> type[_FakeNetworkRunner]:
    """Install a deterministic fake Rust runner for Python-side dispatch tests."""
    _FakeNetworkRunner.instances.clear()
    monkeypatch.setattr(network_module, "_get_rust_engine", lambda: _FakeNetworkRunner)
    return _FakeNetworkRunner


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML manifest through the Python-version appropriate parser."""
    with path.open("rb") as manifest_file:
        return tomllib.load(manifest_file)


__all__ = [
    "builtins",
    "Path",
    "sys",
    "Any",
    "np",
    "pytest",
    "network_module",
    "RateMonitor",
    "SpikeMonitor",
    "StateMonitor",
    "Network",
    "Population",
    "Projection",
    "StepCurrent",
    "tomllib",
    "_MODEL",
    "_FakeNetworkRunner",
    "_install_fake_rust_engine",
    "_load_toml",
]
