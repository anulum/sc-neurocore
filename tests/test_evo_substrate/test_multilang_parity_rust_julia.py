# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust and Julia evolution parity tests

"""Require byte-exact fixed-seed parity between Rust and Julia runners."""

from __future__ import annotations

from typing import Any

pytest_plugins = ["tests.test_evo_substrate.multilang_parity_support"]

JsonObject = dict[str, Any]


def test_rust_julia_bit_exact_best_fitness(
    rust_output: JsonObject, julia_output: JsonObject
) -> None:
    rust_fitness = rust_output["stats_per_generation"][-1]["best_fitness"]
    julia_fitness = julia_output["stats_per_generation"][-1]["best_fitness"]
    assert rust_fitness == julia_fitness, (
        f"Rust best={rust_fitness} Julia best={julia_fitness} — expected byte-exact"
    )


def test_rust_julia_bit_exact_population(rust_output: JsonObject, julia_output: JsonObject) -> None:
    rust_ids = [genome["genome_id"] for genome in rust_output["final_population"]]
    julia_ids = [genome["genome_id"] for genome in julia_output["final_population"]]
    assert rust_ids == julia_ids, (
        f"Final-population genome_ids diverge.\n  Rust:  {rust_ids}\n  Julia: {julia_ids}"
    )


def test_rust_julia_bit_exact_lineage(rust_output: JsonObject, julia_output: JsonObject) -> None:
    assert len(rust_output["lineage"]) == len(julia_output["lineage"])
    for index, (rust_record, julia_record) in enumerate(
        zip(rust_output["lineage"], julia_output["lineage"])
    ):
        assert rust_record["genome_id"] == julia_record["genome_id"], f"lineage[{index}] diverges"


def test_rust_julia_bit_exact_hall_of_fame(
    rust_output: JsonObject, julia_output: JsonObject
) -> None:
    rust_ids = [genome["genome_id"] for genome in rust_output["hall_of_fame"]]
    julia_ids = [genome["genome_id"] for genome in julia_output["hall_of_fame"]]
    assert rust_ids == julia_ids
