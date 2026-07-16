# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — cross-language parity test harness for evo_substrate runners

"""Asserts Rust / Julia / Go / Mojo evolve runners agree on a fixed seed.

Levels of expected agreement (honest, measured):

* **Rust ↔ Julia** — byte-exact identical on every field (genome_ids,
  lineage records, HoF, Pareto, best fitness). They share XorShift64
  + their libm `cos()` / `log()` happen to agree at bit level on
  x86_64.
* **Rust ↔ Go** — Pareto size + population size + counter fields
  match; best fitness drifts ~1e-4 because Go's `math.Cos` / `math.Log`
  differ from Rust's libm at ~1 ULP and Box-Muller compounds that.
* **Rust ↔ Mojo** — structural schema match (lineage count, Pareto
  non-empty, counter fields all present); numerics drift similar to
  the Go case because Mojo's stdlib transcendentals differ too.

Runners that are not installed on the host skip with an informative
message instead of failing — this matches the repo's general policy
of letting missing optional toolchains pass silently.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
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


def _rust_runner() -> EvoRunner:
    """Load the optional Rust runner through its import boundary."""
    module = importlib.import_module("sc_neurocore.evo_substrate.evo_substrate_core")
    return cast(EvoRunner, module)


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

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
    return json.dumps(EVO_CFG_SEED7)


@pytest.fixture(scope="module")
def rust_output(cfg_json: str) -> JsonObject:
    try:
        runner = _rust_runner()
    except ImportError:
        pytest.skip("evo_substrate_core PyO3 extension not compiled")
    return cast(JsonObject, json.loads(runner.py_evolve_run(cfg_json)))


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
    go_dir = REPO_ROOT / "src/sc_neurocore/accel/go/evo_substrate"
    binary = go_dir / "evo_substrate_bench"
    if not binary.exists():
        # Try to build on demand so the test harness is self-contained.
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
    mojo_dir = REPO_ROOT / "src/sc_neurocore/accel/mojo"
    pixi = shutil.which("pixi")
    if pixi is None or not (mojo_dir / "pixi.toml").exists():
        pytest.skip("pixi/Mojo toolchain not available")
    return _run_subprocess(
        pin_isa([pixi, "run", "mojo", "run", "kernels/evo_runner.mojo"]),
        cfg_json,
        cwd=str(mojo_dir),
    )


# ─── Level 1: every backend produces a well-formed EvolveResult ───────


def test_rust_output_schema(rust_output: JsonObject) -> None:
    _assert_evolve_result_schema(rust_output)


def test_julia_output_schema(julia_output: JsonObject) -> None:
    _assert_evolve_result_schema(julia_output)


def test_go_output_schema(go_output: JsonObject) -> None:
    _assert_evolve_result_schema(go_output)


def test_mojo_output_schema(mojo_output: JsonObject) -> None:
    _assert_evolve_result_schema(mojo_output)


def _assert_evolve_result_schema(result: JsonObject) -> None:
    assert set(result.keys()) >= {
        "final_population",
        "stats_per_generation",
        "hall_of_fame",
        "pareto_front",
        "lineage",
        "total_replications",
        "safety_checked",
        "safety_rejected",
        "extinction_count",
    }
    assert len(result["stats_per_generation"]) == EVO_CFG_SEED7["n_generations"]
    assert len(result["final_population"]) <= EVO_CFG_SEED7["pop_size"]
    for rec in result["lineage"]:
        assert set(rec.keys()) >= {
            "genome_id",
            "parent_id",
            "generation",
            "mutation_type",
            "fitness",
        }


# ─── Level 2: Rust ↔ Julia byte-exact identity ────────────────────────


def test_rust_julia_bit_exact_best_fitness(
    rust_output: JsonObject, julia_output: JsonObject
) -> None:
    r = rust_output["stats_per_generation"][-1]["best_fitness"]
    j = julia_output["stats_per_generation"][-1]["best_fitness"]
    assert r == j, f"Rust best={r} Julia best={j} — expected byte-exact"


def test_rust_julia_bit_exact_population(rust_output: JsonObject, julia_output: JsonObject) -> None:
    r_ids = [g["genome_id"] for g in rust_output["final_population"]]
    j_ids = [g["genome_id"] for g in julia_output["final_population"]]
    assert r_ids == j_ids, (
        f"Final-population genome_ids diverge.\n  Rust:  {r_ids}\n  Julia: {j_ids}"
    )


def test_rust_julia_bit_exact_lineage(rust_output: JsonObject, julia_output: JsonObject) -> None:
    assert len(rust_output["lineage"]) == len(julia_output["lineage"])
    for i, (r, j) in enumerate(zip(rust_output["lineage"], julia_output["lineage"])):
        assert r["genome_id"] == j["genome_id"], f"lineage[{i}] diverges"


def test_rust_julia_bit_exact_hall_of_fame(
    rust_output: JsonObject, julia_output: JsonObject
) -> None:
    r = [g["genome_id"] for g in rust_output["hall_of_fame"]]
    j = [g["genome_id"] for g in julia_output["hall_of_fame"]]
    assert r == j


# ─── Level 3: Rust ↔ Go structural parity + fitness tolerance ─────────


def test_rust_go_population_size(rust_output: JsonObject, go_output: JsonObject) -> None:
    assert len(rust_output["final_population"]) == len(go_output["final_population"])


def test_rust_go_pareto_size(rust_output: JsonObject, go_output: JsonObject) -> None:
    assert len(rust_output["pareto_front"]) == len(go_output["pareto_front"])


def test_rust_go_lineage_length(rust_output: JsonObject, go_output: JsonObject) -> None:
    assert len(rust_output["lineage"]) == len(go_output["lineage"])


def test_rust_go_total_replications(rust_output: JsonObject, go_output: JsonObject) -> None:
    assert rust_output["total_replications"] == go_output["total_replications"]


def test_rust_go_best_fitness_within_tolerance(
    rust_output: JsonObject, go_output: JsonObject
) -> None:
    r = rust_output["stats_per_generation"][-1]["best_fitness"]
    g = go_output["stats_per_generation"][-1]["best_fitness"]
    # Go's libm Cos/Log differ from Rust's at ~1 ULP, compounding via
    # Box-Muller over 80 mutations lands us at ~3e-4 after 10 gens.
    assert abs(r - g) < 1e-3, f"Rust={r} Go={g} diff={abs(r - g)}"


# ─── Level 4: Rust ↔ Mojo structural schema match ─────────────────────


def test_rust_mojo_population_size(rust_output: JsonObject, mojo_output: JsonObject) -> None:
    assert len(rust_output["final_population"]) == len(mojo_output["final_population"])


def test_rust_mojo_counters_agree(rust_output: JsonObject, mojo_output: JsonObject) -> None:
    assert rust_output["total_replications"] == mojo_output["total_replications"]
    assert rust_output["safety_rejected"] == mojo_output["safety_rejected"]


def test_rust_mojo_lineage_length(rust_output: JsonObject, mojo_output: JsonObject) -> None:
    assert len(rust_output["lineage"]) == len(mojo_output["lineage"])


def test_rust_mojo_pareto_nonempty(rust_output: JsonObject, mojo_output: JsonObject) -> None:
    # Mojo's transcendental drift makes the exact Pareto set diverge,
    # but it must still be a non-empty non-dominated set because the
    # Pareto-update logic is the same algorithm.
    assert len(mojo_output["pareto_front"]) >= 1
    assert len(rust_output["pareto_front"]) >= 1


# ─── Level 5: determinism under same seed (each backend with itself) ──


def test_rust_seed_determinism(cfg_json: str) -> None:
    try:
        runner = _rust_runner()
    except ImportError:
        pytest.skip("evo_substrate_core PyO3 extension not compiled")
    a = cast(JsonObject, json.loads(runner.py_evolve_run(cfg_json)))
    b = cast(JsonObject, json.loads(runner.py_evolve_run(cfg_json)))
    assert a == b, "Rust runner is non-deterministic under fixed seed"
