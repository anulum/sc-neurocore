# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust and Mojo evolution parity tests

"""Validate structural fixed-seed parity between Rust and Mojo runners."""

from __future__ import annotations

from typing import Any

pytest_plugins = ["tests.test_evo_substrate.multilang_parity_support"]

JsonObject = dict[str, Any]


def test_rust_mojo_population_size(rust_output: JsonObject, mojo_output: JsonObject) -> None:
    assert len(rust_output["final_population"]) == len(mojo_output["final_population"])


def test_rust_mojo_counters_agree(rust_output: JsonObject, mojo_output: JsonObject) -> None:
    assert rust_output["total_replications"] == mojo_output["total_replications"]
    assert rust_output["safety_rejected"] == mojo_output["safety_rejected"]


def test_rust_mojo_lineage_length(rust_output: JsonObject, mojo_output: JsonObject) -> None:
    assert len(rust_output["lineage"]) == len(mojo_output["lineage"])


def test_rust_mojo_pareto_nonempty(rust_output: JsonObject, mojo_output: JsonObject) -> None:
    assert len(mojo_output["pareto_front"]) >= 1
    assert len(rust_output["pareto_front"]) >= 1
