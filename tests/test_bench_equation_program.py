# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The equation program benchmark measures what it says and its receipt is bound

"""Run the equation program benchmark for real and bind its committed receipt and its citations."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest

from tests.benchmark_history_support import assert_committed_source_hashes

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "benchmarks/results/bench_equation_program.json"


def _benchmark() -> ModuleType:
    """Load the benchmark script as a module without running its command line."""
    name = "bench_equation_program"
    specification = importlib.util.spec_from_file_location(
        name, ROOT / "benchmarks/bench_equation_program.py"
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _receipt() -> dict[str, Any]:
    """Return the committed receipt."""
    loaded: dict[str, Any] = json.loads(RECEIPT.read_text(encoding="utf-8"))
    return loaded


def test_a_real_run_times_every_case_and_checks_values_first(tmp_path: Path) -> None:
    """A short real run yields every expression and neuron, with equal values and positive times.

    Parameters
    ----------
    tmp_path : Path
        Directory for the receipt of the short run.
    """
    benchmark = _benchmark()
    output = tmp_path / "receipt.json"
    assert (
        benchmark.main(
            ["--calls", "200", "--steps", "50", "--repeats", "2", "--output", str(output)]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    context = payload["measurement_context"]
    assert context["cpu_model"] and context["host"]
    assert isinstance(context["ram_bytes"], int) and context["ram_bytes"] > 0
    assert context["production_rerun_requirement"]
    assert set(payload["expressions"]) == set(benchmark.EXPRESSIONS)
    assert set(payload["neurons"]) == set(benchmark.NEURONS)
    assert payload["configuration"] == {"calls": 200, "steps": 50, "repeats": 2}
    for entry in payload["expressions"].values():
        assert len(entry["program_ns"]) == len(entry["reference_ns"]) == 2
        assert min(entry["program_ns"] + entry["reference_ns"]) > 0
        assert entry["ratio_of_minima"] == entry["program_min_ns"] / entry["reference_min_ns"]
    assert payload["neurons"]["adaptive_exponential_rk4"]["method"] == "rk4"
    low, high = payload["ratio_of_minima_range"]
    assert 0 < low <= high


def test_a_disagreeing_reference_and_a_non_positive_count_stop_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing is timed when a reference differs from its program, in value or in type.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces one benchmark case for the duration of the test.
    """
    benchmark = _benchmark()
    with pytest.raises(ValueError, match="must be positive"):
        benchmark.build_payload(0, 10, 2)
    monkeypatch.setitem(benchmark.EXPRESSIONS, "threshold", ("v >= v_th", lambda e: 1))
    with pytest.raises(ValueError, match="disagree for threshold"):
        benchmark.measure_expression("threshold", 10, 1)
    monkeypatch.setitem(benchmark.EXPRESSIONS, "threshold", ("v + 1", lambda e: e["v"] + 2))
    with pytest.raises(ValueError, match="disagree for threshold"):
        benchmark.measure_expression("threshold", 10, 1)


def test_the_committed_receipt_is_bound_to_its_sources() -> None:
    """The receipt names every source of the benchmark with the digest it had when recorded."""
    benchmark = _benchmark()
    payload = _receipt()
    assert payload["schema_version"] == "sc-neurocore.equation-program-benchmark.v1"
    assert payload["evidence_class"] == "local_regression"
    assert payload["production_speed_claimed"] is False
    assert payload["all_values_equal_reference"] is True
    assert set(payload["source_sha256"]) == set(benchmark.SOURCE_PATHS)
    assert_committed_source_hashes(RECEIPT, repo_root=ROOT, source_hashes=payload["source_sha256"])


def test_the_documentation_cites_the_ratios_of_the_receipt() -> None:
    """The compiler page and the changelog give the range the committed receipt holds."""
    low, high = _receipt()["ratio_of_minima_range"]
    sentence = f"between {low:.1f} and {high:.1f} times"
    for page in ("docs/api/compiler.md", "docs/CHANGELOG.md"):
        text = " ".join((ROOT / page).read_text(encoding="utf-8").split())
        assert sentence in text, page
        assert "benchmarks/results/bench_equation_program.json" in text, page
