# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolution runner output-schema parity tests

"""Validate the common result schema emitted by every evolution runner."""

from __future__ import annotations

from typing import Any

pytest_plugins = ["tests.test_evo_substrate.multilang_parity_support"]

JsonObject = dict[str, Any]


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
    assert len(result["stats_per_generation"]) == 10
    assert len(result["final_population"]) <= 16
    for record in result["lineage"]:
        assert set(record.keys()) >= {
            "genome_id",
            "parent_id",
            "generation",
            "mutation_type",
            "fitness",
        }


def test_rust_output_schema(rust_output: JsonObject) -> None:
    _assert_evolve_result_schema(rust_output)


def test_julia_output_schema(julia_output: JsonObject) -> None:
    _assert_evolve_result_schema(julia_output)


def test_go_output_schema(go_output: JsonObject) -> None:
    _assert_evolve_result_schema(go_output)


def test_mojo_output_schema(mojo_output: JsonObject) -> None:
    _assert_evolve_result_schema(mojo_output)
