# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC optimizer engine-binding contracts

"""Public contracts for the SC optimizer PyO3 functions."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()
import sc_neurocore_engine as engine


def test_exported_function_names_and_search_signature_are_stable() -> None:
    """Preserve the exported identities and supported search defaults."""
    assert engine.py_opt_sa_search.__name__ == "py_opt_sa_search"
    assert engine.py_opt_extract_pareto.__name__ == "py_opt_extract_pareto"
    assert engine.py_opt_sa_search.__text_signature__ == (
        "(mac_counts, weights, max_luts, max_power, max_latency=0, t_init=1.0, "
        "t_min=0.001, alpha=0.95, max_iter=2000, seed=42)"
    )


def test_simulated_annealing_search_is_seeded_and_reports_layer_resources() -> None:
    """Return reproducible configurations with one resource row per layer."""
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
    """Distinguish an infeasible budget from a malformed search request."""
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
    """Retain the complete columns of every non-dominated point."""
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


_SEARCH_PROGRAM = """
import hashlib
import json
from pathlib import Path
import sys
if sys.platform == 'linux':
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import sc_neurocore_engine as engine
import sc_neurocore_engine.sc_neurocore_engine as native

origins, mac_counts, weights, message = map(json.loads, sys.argv[1:])
for module in (engine, native):
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve())
    assert str(path) == origins[module.__name__]['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == origins[module.__name__]['sha256']
assert engine.py_opt_sa_search is native.py_opt_sa_search
healthy = native.py_opt_sa_search([10, 20], [1.0, 1.0], 100000, 1000.0, max_iter=10)
assert healthy['feasible'] and healthy['best_config'] == [29, 14]
assert healthy['best_score'] == 0.99736328125
before = json.dumps([mac_counts, weights])
for iterations in (0, 10):
    try:
        engine.py_opt_sa_search(mac_counts, weights, 100000, 1000.0, max_iter=iterations)
    except ValueError as error:
        assert str(error) == message
    else:
        raise AssertionError('missing optimizer inputs accepted')
    assert json.dumps([mac_counts, weights]) == before
    assert native.py_opt_sa_search([10, 20], [1.0, 1.0], 100000, 1000.0, max_iter=10) == healthy
for module in (engine, native):
    assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == origins[module.__name__]['sha256']
print(json.dumps({'refusals': 2, 'caller_inputs_unchanged': True, 'healthy_retry': True}))
"""


@pytest.mark.parametrize(
    ("mac_counts", "weights", "message"),
    [
        ([], [], "search requires at least one layer"),
        ([], [1.0], "search requires at least one layer"),
        ([10], [], "weights must cover every layer"),
        ([10, 20], [1.0], "weights must cover every layer"),
    ],
)
def test_missing_search_inputs_raise_value_error_and_allow_retry(
    mac_counts: list[int], weights: list[float], message: str
) -> None:
    """Refuse malformed searches in an installed consumer without process abort.

    Parameters
    ----------
    mac_counts
        Layer MAC counts supplied to the public search function.
    weights
        Scoring weights, including the missing-layer cases.
    message
        Exact authored refusal before any candidate construction.
    """
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _SEARCH_PROGRAM,
            json.dumps(installed_engine_origins()),
            json.dumps(mac_counts),
            json.dumps(weights),
            json.dumps(message),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "refusals": 2,
        "caller_inputs_unchanged": True,
        "healthy_retry": True,
    }


def test_search_retains_trailing_weight_normalization_and_zero_iterations() -> None:
    """Keep accepted trailing weights and zero-iteration initial configurations."""
    initial = engine.py_opt_sa_search([10, 20], [1.0, 1.0], 100_000, 1000.0, max_iter=0)
    trailing = engine.py_opt_sa_search([10, 20], [1.0, 1.0, 2.0], 100_000, 1000.0, max_iter=0)
    assert initial["feasible"] and trailing["feasible"]
    assert trailing["best_config"] == initial["best_config"]
    assert trailing["best_score"] == initial["best_score"] / 2.0
    for field in ("layer_luts", "layer_power", "layer_accuracy"):
        assert trailing[field] == initial[field]
    assert trailing["pareto_score"] == initial["pareto_score"] == []
