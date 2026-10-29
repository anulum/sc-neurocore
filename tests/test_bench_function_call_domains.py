# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Function-domain benchmark CLI and measurement refusals
"""Execute the real benchmark and reject invalid or inconsistent evidence."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import bench_function_call_domains as bench
from sc_neurocore.compiler import function_call_domains as domain_module
from sc_neurocore.compiler.function_call_domains import FunctionCallContract
from sc_neurocore.neurons.equation_builder import EquationNeuron


def test_real_cli_measures_every_case_with_source_and_cache_scope(tmp_path: Path) -> None:
    """The parent launches real workers and writes complete, reproducible contracts.

    Parameters
    ----------
    tmp_path : Path
        Output directory for this short run.
    """
    output = tmp_path / "domains.json"
    assert bench.main(["--calls", "2", "--repeats", "2", "--output", str(output)]) == 0
    data = json.loads(output.read_text())
    assert data["schema_version"] == "sc-neurocore.function-call-domain-benchmark.v1"
    assert data["production_speed_claimed"] is False
    assert data["source_unchanged_during_capture"] is True
    assert set(data["cases"]) == set(bench.CASES)
    assert len(data["source_head_at_capture"]) == 40
    assert set(data["source_sha256"]) == set(bench.SOURCE_PATHS)
    for path, digest in data["source_sha256"].items():
        assert hashlib.sha256((bench.ROOT / path).read_bytes()).hexdigest() == digest
    assert data["measurement_context"]["cpu_model"]
    assert data["measurement_context"]["production_rerun_requirement"]
    for name, result in data["cases"].items():
        assert result["first_call_ns"] > 0
        assert len(result["cached_ns_per_call"]) == 2
        assert min(result["cached_ns_per_call"]) > 0
        assert result["all_repeats_equal"] is True
        contracts = result["contracts"]
        expected = bench.measure_case(name, 1, 1)
        assert contracts == expected["contracts"]
        assert result["contract_sha256"] == expected["contract_sha256"]
        assert all(call["accuracy"]["codes_evaluated"] > 0 for call in contracts)
    paired = data["cases"]["paired_exponentials"]["contracts"]
    assert [call["accuracy"]["codes_evaluated"] for call in paired] == [256, 64]
    events = data["cases"]["exponential_euler_events"]["contracts"]
    assert any(call["implicit"] and call["function"] == "exprel" for call in events)


@pytest.mark.parametrize(("calls", "repeats"), [(0, 1), (1, 0), (-1, 1), (1, -1)])
def test_nonpositive_counts_do_not_produce_evidence(calls: int, repeats: int) -> None:
    """Both worker and parent reject empty measurement runs.

    Parameters
    ----------
    calls, repeats : int
        One invalid count and one positive count.
    """
    with pytest.raises(ValueError, match="must be positive"):
        bench.measure_case(bench.CASES[0], calls, repeats)
    with pytest.raises(ValueError, match="must be positive"):
        bench.build_payload(calls, repeats)


def test_worker_cli_and_unknown_model(capsys: pytest.CaptureFixture[str]) -> None:
    """Worker output is parseable JSON; unknown cases cannot silently fall back.

    Parameters
    ----------
    capsys : CaptureFixture
        Captures the actual worker protocol output.
    """
    assert (
        bench.main(["--measure-case", "paired_exponentials", "--calls", "1", "--repeats", "1"]) == 0
    )
    data = json.loads(capsys.readouterr().out)
    assert data["first_call_ns"] > 0 and len(data["contracts"]) == 2
    with pytest.raises(KeyError):
        bench.measure_case("unknown", 1, 1)
    with pytest.raises(SystemExit) as error:
        bench.main(["--measure-case", "unknown"])
    assert error.value.code == 2


@pytest.mark.parametrize("expression", ["I", "exp(I,None)"])
def test_empty_or_refused_contract_is_not_timing_evidence(
    monkeypatch: pytest.MonkeyPatch, expression: str
) -> None:
    """Real collector refusals cannot be mistaken for a successful analysis run.

    Parameters
    ----------
    monkeypatch : MonkeyPatch
        Supplies an authored model outside the benchmark's qualified cases.
    expression : str
        An expression with no calls, or a real hardware call refusal.
    """
    monkeypatch.setattr(bench, "_model", lambda _: EquationNeuron({"v": expression}, method="map"))
    with pytest.raises(ValueError, match="no complete call-domain evidence"):
        bench.measure_case(bench.CASES[0], 1, 1)


def test_changed_contract_refuses_the_cached_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deliberately changed source location is detected after a real first call.

    Parameters
    ----------
    monkeypatch : MonkeyPatch
        Injects a contract corruption after the initial real collector result.
    """
    original = domain_module.function_call_contracts
    invocations = 0

    def changed(
        model: EquationNeuron, width: int, fraction: int
    ) -> tuple[FunctionCallContract, ...]:
        nonlocal invocations
        invocations += 1
        result = original(model, width, fraction)
        return result if invocations == 1 else (replace(result[0], column=999), *result[1:])

    monkeypatch.setattr(domain_module, "function_call_contracts", changed)
    with pytest.raises(ValueError, match="changed between repetitions"):
        bench.measure_case(bench.CASES[0], 1, 1)


def test_import_from_a_different_checkout_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Source pins cannot describe an unrelated installed collector.

    Parameters
    ----------
    monkeypatch : MonkeyPatch
        Changes the observed origin while retaining the real callable.
    """
    monkeypatch.setattr(domain_module, "__file__", str(bench.ROOT / "wrong.py"))
    with pytest.raises(RuntimeError, match="does not belong"):
        bench.build_payload(1, 1)


def test_changed_source_snapshot_refuses_real_worker_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timings are discarded when the final source snapshot differs.

    Parameters
    ----------
    monkeypatch : MonkeyPatch
        Corrupts one final digest after all real child measurements.
    """
    original = bench._source_hashes
    invocations = 0

    def changed() -> dict[str, str]:
        nonlocal invocations
        invocations += 1
        snapshot = original()
        if invocations > 1:
            snapshot[bench.SOURCE_PATHS[0]] = "0" * 64
        return snapshot

    monkeypatch.setattr(bench, "_source_hashes", changed)
    with pytest.raises(ValueError, match="sources changed"):
        bench.build_payload(1, 1)


def test_recorded_timings_are_the_figures_published_in_the_guide() -> None:
    """The measured receipt, blocking gate and guide describe the same four cases."""
    path = "benchmarks/results/bench_function_call_domains.json"
    data = json.loads((bench.ROOT / path).read_text())
    guide = (bench.ROOT / "docs/guides/function_call_domains.md").read_text()
    manifest = json.loads((bench.ROOT / "benchmarks/benchmark_regression_gates.json").read_text())
    (gate,) = [entry for entry in manifest["gates"] if entry["artefact"] == path]
    assert data["configuration"] == {"calls": 100, "repeats": 7, "data_width": 8, "fraction": 4}
    assert set(data["reference_dependency_versions"]) == {"numpy", "scipy", "sympy"}
    assert all(data["reference_dependency_versions"].values())
    assert data["production_speed_claimed"] is False
    assert data["source_unchanged_during_capture"] is True
    assert set(data["cases"]) == set(bench.CASES)
    assert gate["source_hashes"] == {p: f"source_sha256.{p}" for p in data["source_sha256"]}
    for name in bench.CASES:
        result = data["cases"][name]
        assert len(result["cached_ns_per_call"]) == 7
        assert result["first_call_ns"] > 0 and min(result["cached_ns_per_call"]) > 0
        assert result["all_repeats_equal"] is True
        assert (
            f"| {result['first_call_ns'] / 1e6:.2f} | {result['cached_median_ns_per_call'] / 1e3:.2f} |"
            in guide
        )
