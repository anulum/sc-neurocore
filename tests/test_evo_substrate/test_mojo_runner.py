# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo evo_runner side-validation via Python driver

"""Python-side validation of the Mojo evolve runner.

The runner is exercised as a subprocess (the same dispatch pattern Python
uses in production) and the suite asserts invariants on its JSON output. This mirrors what
a `go test` / `Test.jl` unit suite would cover.

Skipped when pixi / Mojo toolchain is missing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from sc_neurocore.accel.mojo.isa_baseline import pin_isa

JsonObject = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MOJO_DIR = REPO_ROOT / "src/sc_neurocore/accel/mojo"
RUNNER = MOJO_DIR / "kernels/evo_runner.mojo"
_PIXI_PATH = shutil.which("pixi")
PIXI = Path(_PIXI_PATH) if _PIXI_PATH is not None else None


def _minimal_cfg(seed: int, pop: int = 8, gens: int = 3) -> str:
    cfg = {
        "seed": seed,
        "pop_size": pop,
        "n_generations": gens,
        "elitism": 1,
        "survival_fraction": 0.5,
        "tournament_size": 3,
        "crossover_prob": 0.3,
        "max_age": 20,
        "hall_of_fame_size": 5,
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
    return json.dumps(cfg)


def _run_mojo(cfg: str) -> JsonObject:
    # Mojo JIT-compiles the runner on every invocation (~1 min on
    # cold pixi cache, faster once warm). 900 s gives comfortable margin
    # for the slowest cold run without hiding a genuine hang.
    assert PIXI is not None
    proc = subprocess.run(
        pin_isa([str(PIXI), "run", "mojo", "run", "kernels/evo_runner.mojo"]),
        input=cfg,
        capture_output=True,
        text=True,
        cwd=str(MOJO_DIR),
        timeout=900,
    )
    if proc.returncode != 0:
        pytest.fail(f"Mojo runner failed: {proc.stderr[:400]}")
    return cast(JsonObject, json.loads(proc.stdout))


pytestmark = pytest.mark.skipif(
    PIXI is None or not (MOJO_DIR / "pixi.toml").exists(),
    reason="pixi/Mojo toolchain not available",
)


def test_mojo_runner_produces_valid_schema() -> None:
    r = _run_mojo(_minimal_cfg(seed=7))
    required = {
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
    assert set(r.keys()) >= required


def test_mojo_runner_generation_count_matches_config() -> None:
    r = _run_mojo(_minimal_cfg(seed=7, gens=5))
    assert len(r["stats_per_generation"]) == 5


def test_mojo_runner_lineage_records_have_full_schema() -> None:
    r = _run_mojo(_minimal_cfg(seed=7))
    assert len(r["lineage"]) > 0
    for rec in r["lineage"]:
        assert set(rec.keys()) >= {
            "genome_id",
            "parent_id",
            "generation",
            "mutation_type",
            "fitness",
        }


def test_mojo_runner_pareto_front_is_non_empty_on_default_fitness() -> None:
    r = _run_mojo(_minimal_cfg(seed=7))
    # The default parametric fitness always produces at least one
    # non-dominated organism, so the Pareto front must be non-empty.
    assert len(r["pareto_front"]) >= 1


def test_mojo_runner_genome_ids_are_12_hex_chars() -> None:
    r = _run_mojo(_minimal_cfg(seed=7))
    for g in r["final_population"]:
        assert len(g["genome_id"]) == 12, f"bad id: {g['genome_id']!r}"
        int(g["genome_id"], 16)  # must be valid hex


def test_mojo_runner_is_deterministic_under_same_seed() -> None:
    # Minimal workload — Mojo subprocess JIT compile dominates (~7 min
    # per call on cold pixi cache). Two runs × small pop/gens is still
    # a valid parity check because the deterministic XorShift64 state
    # evolution is independent of population size.
    a = _run_mojo(_minimal_cfg(seed=11, pop=2, gens=1))
    b = _run_mojo(_minimal_cfg(seed=11, pop=2, gens=1))
    a_ids = [g["genome_id"] for g in a["final_population"]]
    b_ids = [g["genome_id"] for g in b["final_population"]]
    assert a_ids == b_ids


def test_mojo_runner_counters_are_monotonic() -> None:
    r = _run_mojo(_minimal_cfg(seed=7, gens=5))
    assert r["total_replications"] >= len(r["lineage"]) - 8  # minus seed records
    assert r["safety_rejected"] <= r["safety_checked"]
