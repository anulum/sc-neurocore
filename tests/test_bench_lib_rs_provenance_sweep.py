# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — committed benchmark-evidence crate-root provenance sweep

"""Repository sweep keeping benchmark evidence off shared engine aggregators."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tool() -> Any:
    """Load the crate-root provenance gate module by path (tools/ is not a package)."""
    tool_path = REPO_ROOT / "tools" / "lib_rs_provenance_gate.py"
    spec = importlib.util.spec_from_file_location("lib_rs_provenance_gate", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_tool()


def test_all_committed_benchmark_evidence_has_no_stale_crate_root_binding() -> None:
    """Reject any legacy crate-root binding whose digest has drifted from live."""
    failures = gate.evaluate(repo_root=REPO_ROOT)
    assert failures == [], "stale engine/src/lib.rs provenance in: " + ", ".join(
        f"{failure.artefact} (recorded {failure.recorded[:12]}, live {failure.expected[:12]})"
        for failure in failures
    )


def test_benchmarks_use_model_owned_sources_not_shared_engine_aggregators() -> None:
    """Keep evidence stable across unrelated engine facade decomposition."""
    binders = {
        path.relative_to(REPO_ROOT).as_posix()
        for path, _ in gate.iter_crate_root_bindings(REPO_ROOT / "benchmarks")
    }
    assert binders == set()

    shared_aggregators = ("engine/src/lib.rs", "engine/src/pyo3_neurons.rs")
    offenders = {
        path.relative_to(REPO_ROOT).as_posix(): aggregate
        for path in sorted((REPO_ROOT / "benchmarks").glob("bench_*.py"))
        for aggregate in shared_aggregators
        if aggregate in path.read_text(encoding="utf-8")
    }
    assert offenders == {}
