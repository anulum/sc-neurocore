# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC optimizer engine-binding contracts

"""Public contracts for the SC optimizer PyO3 functions."""

from __future__ import annotations

import sc_neurocore_engine as engine


def test_exported_function_names_and_search_signature_are_stable() -> None:
    assert engine.py_opt_sa_search.__name__ == "py_opt_sa_search"
    assert engine.py_opt_extract_pareto.__name__ == "py_opt_extract_pareto"
    assert engine.py_opt_sa_search.__text_signature__ == (
        "(mac_counts, weights, max_luts, max_power, max_latency=0, t_init=1.0, "
        "t_min=0.001, alpha=0.95, max_iter=2000, seed=42)"
    )


def test_simulated_annealing_search_is_seeded_and_reports_layer_resources() -> None:
    arguments = ([10, 20], [1.0, 1.0], 100_000, 1000.0, 0, 1.0, 0.001, 0.95, 100, 42)
    first = engine.py_opt_sa_search(*arguments)
    second = engine.py_opt_sa_search(*arguments)

    assert first == second
    assert first["backend"] == "rust"
    assert first["feasible"] is True
    assert len(first["best_config"]) == 2
    assert len(first["layer_luts"]) == 2
    assert len(first["layer_power"]) == 2
    assert len(first["layer_accuracy"]) == 2


def test_infeasible_search_has_minimal_fail_closed_result() -> None:
    result = engine.py_opt_sa_search(
        [10],
        [1.0],
        0,
        0.0,
        0,
        1.0,
        0.001,
        0.95,
        10,
        42,
    )
    assert result == {"backend": "rust", "feasible": False}


def test_pareto_extraction_preserves_selected_values() -> None:
    result = engine.py_opt_extract_pareto(
        [100, 200, 150],
        [1.0, 0.5, 0.8],
        [0.9, 0.95, 0.85],
    )
    assert result == {
        "backend": "rust",
        "indices": [0, 1, 2],
        "luts": [100, 200, 150],
        "power": [1.0, 0.5, 0.8],
        "score": [0.9, 0.95, 0.85],
    }
