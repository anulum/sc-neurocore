# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolution runner parity fixtures and schema assertions

"""Shared fixtures for cross-language evolution-runner parity tests."""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Protocol, cast

import pytest

from sc_neurocore.accel.mojo.isa_baseline import pin_isa

JsonObject = dict[str, Any]


class EvoRunner(Protocol):
    """Typed boundary for the optional PyO3 evolution runner."""

    def py_evolve_run(self, config_json: str) -> str:
        """Run one evolution configuration and return JSON."""
        ...


def rust_runner() -> EvoRunner:
    """Load the optional Rust runner through its import boundary."""
    module = importlib.import_module("sc_neurocore.evo_substrate.evo_substrate_core")
    return cast(EvoRunner, module)


REPO_ROOT = Path(__file__).resolve().parents[2]

EVO_CFG_SEED7: JsonObject = {
    "seed": 7,
    "pop_size": 16,
    "n_generations": 10,
    "elitism": 1,
    "survival_fraction": 0.5,
    "tournament_size": 3,
    "crossover_prob": 0.3,
    "max_age": 20,
    "hall_of_fame_size": 10,
    "stagnation_gens": 10,
    "extinction_kill_fraction": 0.9,
    "mutation": {
        "point_rate": 0.2,
        "point_sigma": 0.05,
        "structural_rate": 0.05,
        "duplication_rate": 0.01,
        "swap_rate": 0.02,
        "max_neurons": 1024,
        "min_neurons": 4,
    },
    "fitness": {
        "accuracy_bias": 0.5,
        "accuracy_neuron_coef": 0.01,
        "w_accuracy": 0.5,
        "w_energy": 0.3,
        "w_latency": 0.2,
    },
    "safety_bounds": {
        "max_neurons": 1024,
        "min_neurons": 4,
        "max_layers": 16,
        "max_bitstream": 4096,
        "min_bitstream": 32,
        "max_connectivity": 1.0,
    },
    "industrial_mode": True,
}


@pytest.fixture(scope="module")
def cfg_json() -> str:
    """Return the fixed-seed evolution configuration as JSON."""
    return json.dumps(EVO_CFG_SEED7)


@pytest.fixture(scope="module")
def rust_output(cfg_json: str) -> JsonObject:
    """Run the Rust evolution backend for the shared configuration."""
    try:
        runner = rust_runner()
    except ImportError:
        pytest.skip("evo_substrate_core PyO3 extension not compiled")
    return cast(JsonObject, json.loads(runner.py_evolve_run(cfg_json)))


@pytest.fixture(scope="module")
def rust_runner_backend() -> EvoRunner:
    """Load the Rust evolution runner or skip when its extension is absent."""
    try:
        return rust_runner()
    except ImportError:
        pytest.skip("evo_substrate_core PyO3 extension not compiled")


def _run_subprocess(cmd: list[str], cfg_json: str, cwd: str | None = None) -> JsonObject:
    proc = subprocess.run(
        cmd,
        input=cfg_json,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=600,
    )
    if proc.returncode != 0:
        pytest.fail(f"{cmd[0]} runner failed: {proc.stderr[:500]}")
    return cast(JsonObject, json.loads(proc.stdout))


@pytest.fixture(scope="module")
def julia_output(cfg_json: str) -> JsonObject:
    """Run the Julia evolution backend for the shared configuration."""
    julia = Path.home() / ".juliaup" / "bin" / "julia"
    if not julia.exists():
        pytest.skip(f"julia binary not found at {julia}")
    script = REPO_ROOT / "src/sc_neurocore/accel/julia/evo_substrate/evo_runner.jl"
    return _run_subprocess(
        [str(julia), f"--project={script.parent}", str(script)],
        cfg_json,
    )


@pytest.fixture(scope="module")
def go_output(cfg_json: str) -> JsonObject:
    """Run the Go evolution backend for the shared configuration."""
    go_dir = REPO_ROOT / "src/sc_neurocore/accel/go/evo_substrate"
    binary = go_dir / "evo_substrate_bench"
    if not binary.exists():
        try:
            subprocess.run(
                ["go", "build", "-o", str(binary), "."],
                cwd=str(go_dir),
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Go toolchain not available or build failed")
    return _run_subprocess([str(binary), "--runner"], cfg_json)


@pytest.fixture(scope="module")
def mojo_output(cfg_json: str) -> JsonObject:
    """Run the Mojo evolution backend for the shared configuration."""
    mojo_dir = REPO_ROOT / "src/sc_neurocore/accel/mojo"
    pixi = shutil.which("pixi")
    if pixi is None or not (mojo_dir / "pixi.toml").exists():
        pytest.skip("pixi/Mojo toolchain not available")
    return _run_subprocess(
        pin_isa([pixi, "run", "mojo", "run", "kernels/evo_runner.mojo"]),
        cfg_json,
        cwd=str(mojo_dir),
    )
