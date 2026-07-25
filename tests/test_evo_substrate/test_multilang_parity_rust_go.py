# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust and Go evolution parity tests

"""Validate structural parity and bounded numeric drift for Rust and Go."""

from __future__ import annotations

from typing import Any

pytest_plugins = ["tests.test_evo_substrate.multilang_parity_support"]

JsonObject = dict[str, Any]


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
    rust_fitness = rust_output["stats_per_generation"][-1]["best_fitness"]
    go_fitness = go_output["stats_per_generation"][-1]["best_fitness"]
    assert abs(rust_fitness - go_fitness) < 1e-3, (
        f"Rust={rust_fitness} Go={go_fitness} diff={abs(rust_fitness - go_fitness)}"
    )
