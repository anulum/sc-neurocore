# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python coverage threshold cross-surface contract

"""Keep local, preflight, and hosted Python coverage gates at exact closure."""

from __future__ import annotations

from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]  # Python 3.10 fallback


_ROOT = Path(__file__).resolve().parents[1]


def _cov_thresholds(path: Path) -> tuple[int, ...]:
    """Return explicit pytest-cov thresholds from one owned command surface."""
    text = path.read_text(encoding="utf-8")
    return tuple(int(value) for value in re.findall(r"--cov-fail-under(?:=|\s+)(\d+)", text))


def test_python_coverage_threshold_is_exact_closure() -> None:
    """The canonical config drives preflight and must reject any missing statement."""
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["tool"]["coverage"]["report"]["fail_under"] == 100


def test_python_coverage_reports_repository_source_only() -> None:
    """Extracted wheel copies measured in child interpreters are not part of the gate.

    The copies are recorded under temporary paths by package name. Reporting them
    fails once their directory is gone, and their lines are not repository source.
    """
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["tool"]["coverage"]["run"]["source"] == ["sc_neurocore"]
    assert data["tool"]["coverage"]["report"]["include"] == ["src/sc_neurocore/*"]
    assert "omit" not in data["tool"]["coverage"]["report"]
    assert "ignore_errors" not in data["tool"]["coverage"]["report"]


def test_python_coverage_keeps_isolated_workers_and_watchdog_exits() -> None:
    """Native Python workers and immediate watchdog exits retain process-specific data."""
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert set(data["tool"]["coverage"]["run"]["patch"]) >= {"subprocess", "_exit"}
    assert data["tool"]["coverage"]["run"]["parallel"] is True
    assert "src/sc_neurocore" in data["tool"]["coverage"]["run"]["source_dirs"]
    assert data["tool"]["coverage"]["run"]["relative_files"] is True


def test_primary_coverage_requires_native_windows_refusal_evidence() -> None:
    """The aggregate cannot pass without the actual Windows test job and its data."""
    import yaml

    workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text())
    native = workflow["jobs"]["shd-non-linux-refusal"]
    assert native["runs-on"] == "windows-latest"
    assert any(
        "tests/test_accel_shd_platform.py" in step.get("run", "") for step in native["steps"]
    )
    upload = next(
        step
        for step in native["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["include-hidden-files"] is True
    assert "shd-non-linux-refusal" in workflow["jobs"]["test"]["needs"]
    union = workflow["jobs"]["coverage-union"]
    assert "shd-non-linux-refusal" in union["needs"]
    download = next(
        step
        for step in union["steps"]
        if step.get("name") == "Retrieve actual Windows refusal coverage"
    )
    assert "if" not in download
    assert download["with"]["name"] == upload["with"]["name"]
    aggregate = next(step for step in union["steps"] if step.get("name") == "Coverage union gate")
    assert aggregate["env"]["SCN_EXTRA_COVERAGE_DIR"] == download["with"]["path"]
    runner = (_ROOT / "tools/run_full_cov.sh").read_text()
    assert 'coverage combine --append --keep "$SCN_EXTRA_COVERAGE_DIR"' in runner
    assert runner.index("coverage combine --append") < runner.index("coverage report --skip-empty")


def test_explicit_local_and_hosted_thresholds_match_canonical_gate() -> None:
    """Makefile and primary hosted CI must not weaken the canonical threshold."""
    assert _cov_thresholds(_ROOT / "Makefile") == (100,)
    assert _cov_thresholds(_ROOT / ".github/workflows/ci.yml") == (100, 100)


def test_hosted_full_suite_uses_memory_reclaiming_batches() -> None:
    """Run every matrix leg in bounded batches without weakening coverage."""
    workflow_path = _ROOT / ".github/workflows/ci.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")
    start = workflow_text.index("- name: Test + coverage")
    end = workflow_text.index("\n      - name:", start + 1)
    step = workflow_text[start:end]

    assert step.count("bash tools/run_full_cov.sh --batch-size=384") == 2
    assert step.count('--shard="${{ matrix.shard-index }}/$SCN_CI_SHARD_COUNT"') == 2
    assert step.count('--timings="$RUNNER_TEMP/pytest-timings/pytest-durations.json"') == 2
    assert "--batch-size=384 --no-cov" in step
    assert "--cov-fail-under" not in step
    assert "pytest -n auto" not in step

    try:
        import yaml
    except ModuleNotFoundError:  # pragma: no cover - PyYAML ships with the dev env.
        import pytest

        pytest.skip("PyYAML is not installed")
    workflow = yaml.safe_load(workflow_text)
    # Every leg stays below the hosted runner's six-hour limit; coverage shards have five.
    assert workflow["jobs"]["test"]["timeout-minutes"] == "${{ matrix.coverage && 300 || 240 }}"


def test_sharded_coverage_is_gated_once_on_the_complete_union() -> None:
    """Coverage shards cover the suite once; the required check gates their union.

    A skipped union job would report success to branch protection, so it must run
    after any outcome and refuse unless every shard published its data.
    """
    try:
        import yaml
    except ModuleNotFoundError:  # pragma: no cover - PyYAML ships with the dev env.
        import pytest

        pytest.skip("PyYAML is not installed")
    workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    test = workflow["jobs"]["test"]
    matrix = test["strategy"]["matrix"]
    include = matrix["include"]
    assert matrix["python-minor"] == ["3.11", "3.12", "3.13", "3.14"]
    assert matrix["shard-index"] == list(range(1, 9))
    count = len(matrix["shard-index"])
    assert int(workflow["env"]["SCN_CI_SHARD_COUNT"]) == count
    coverage = [leg for leg in include if leg.get("coverage") is True]
    assert coverage == [{"python-minor": "3.12", "python-version": "3.12.16", "coverage": True}]
    assert "matrix.shard-index" in test["name"]
    assert "matrix.coverage" in test["name"]

    stage, preserve = test["steps"][-2:]
    assert stage["if"] == preserve["if"] == "matrix.coverage"
    assert ".coverage.shard-${{ matrix.shard-index }}" in stage["run"]
    assert preserve["with"]["name"] == "coverage-data-shard-${{ matrix.shard-index }}"
    assert preserve["with"]["include-hidden-files"] is True
    assert preserve["with"]["if-no-files-found"] == "error"

    union = workflow["jobs"]["coverage-union"]
    assert union["name"] == "test (3.12)"
    assert union["if"] == "${{ !cancelled() }}"
    assert "test" in union["needs"]
    retrieve = next(
        step for step in union["steps"] if step.get("name") == "Retrieve every coverage shard"
    )
    assert retrieve["with"]["pattern"] == "coverage-data-shard-*"
    assert retrieve["with"]["merge-multiple"] is True
    gate = next(step for step in union["steps"] if step.get("name") == "Coverage union gate")
    assert '--combine-from="$RUNNER_TEMP/coverage-shards"' in gate["run"]
    assert '--expect-shards="$SCN_CI_SHARD_COUNT"' in gate["run"]
    assert "--cov-fail-under=100" in gate["run"]
    assert "coverage-union" in workflow["jobs"]["build"]["needs"]
    readiness = union["steps"][0]
    assert readiness["env"] == {
        "TEST_RESULT": "${{ needs.test.result }}",
        "WINDOWS_RESULT": "${{ needs.shd-non-linux-refusal.result }}",
    }
    assert readiness["run"] == 'test "$TEST_RESULT" = success && test "$WINDOWS_RESULT" = success'
    compatibility = workflow["jobs"]["compatibility-union"]
    assert compatibility["name"] == "test (${{ matrix.python-minor }})"
    assert compatibility["needs"] == ["test"]
    assert compatibility["if"] == "${{ !cancelled() }}"
    assert compatibility["strategy"]["matrix"]["python-minor"] == ["3.11", "3.13", "3.14"]
    assert compatibility["steps"][0]["env"] == {"TEST_RESULT": "${{ needs.test.result }}"}
    assert compatibility["steps"][0]["run"] == 'test "$TEST_RESULT" = success'


def test_hosted_shards_share_current_abi_wheels_and_one_timing_snapshot() -> None:
    """Every shard consumes its ABI's current artifact and the same run's scheduling hints."""
    import yaml

    workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text())
    jobs = workflow["jobs"]
    test = jobs["test"]
    builders = jobs["test-engine-wheels"]
    assert builders["strategy"]["matrix"]["python-version"] == [
        row["python-version"]
        for row in test["strategy"]["matrix"]["include"]
        if "python-version" in row
    ]
    assert builders["runs-on"] == test["runs-on"] == "ubuntu-24.04"
    assert {"test-engine-wheels", "test-timing-baseline"} <= set(test["needs"])
    build = next(
        step
        for step in builders["steps"]
        if step.get("name") == "Build the current engine once per Python ABI"
    )
    assert 'maturin build --release --locked --out "$RUNNER_TEMP/engine-wheel"' in build["run"]
    producer = builders["steps"][-1]
    consumer = next(
        step
        for step in test["steps"]
        if step.get("name") == "Retrieve this run's ABI-matching engine"
    )
    assert consumer["with"]["name"] == producer["with"]["name"]
    install = next(step for step in test["steps"] if step.get("name") == "Install package")
    assert install["env"]["SC_NEUROCORE_CI_ENGINE_WHEEL_DIR"] == consumer["with"]["path"]
    cache = next(step for step in builders["steps"] if step.get("id") == "engine-wheel")
    assert "restore-keys" not in cache["with"]
    for source in ("engine/src/**", "bridge/vendor/**", "Cargo.lock", "requirements/maturin.txt"):
        assert source in cache["with"]["key"]
    timing_consumer = next(
        step
        for step in test["steps"]
        if step.get("name") == "Retrieve this run's shared timing baseline"
    )
    assert (
        timing_consumer["with"]["name"] == jobs["test-timing-baseline"]["steps"][-1]["with"]["name"]
    )
    assert jobs["test-timings"]["if"] == "${{ !cancelled() }}"
    assert jobs["test-timings"]["needs"] == ["test"]


def test_batched_runner_collects_every_pytest_module_and_unions_coverage() -> None:
    """The OOM repair must retain complete discovery and the exact coverage gate."""
    runner = (_ROOT / "tools/run_full_cov.sh").read_text(encoding="utf-8")
    assert "find tests -type f" in runner
    assert "-name 'test_*.py'" in runner
    assert "-name '*_test.py'" in runner
    assert "--cov=sc_neurocore --cov-append --cov-report=" in runner
    assert "--cov-fail-under=0" in runner
    assert "DEFAULT_COV_FAIL_UNDER=100" in runner
    assert '--fail-under="$cov_fail_under"' in runner
    assert "python -m coverage xml -o coverage.xml" in runner
    assert "test-results-batch-%03d.xml" in runner
    assert '! $coverage && [[ "$path" == "tests/test_catalogue_formal_smoke.py" ]]' in runner
    assert "index = shard_index - 1; index < ${#test_files[@]}; index += shard_count" in runner
    assert "((${#shard_files[@]} != expect_shards))" in runner


def test_nagumo_sato_and_sc_map_exact_coverage_and_parity_is_hosted() -> None:
    """Both preserved map identities need branch and backend custody."""
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index(
        "- name: Nagumo-Sato and SC adaptive-map exact coverage and backend parity"
    )
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.coverage && matrix.shard-index == 4" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "src/sc_neurocore/neurons/models/nagumo_sato_map_neuron.py" in step
    assert "src/sc_neurocore/neurons/models/sc_adaptive_threshold_map_neuron.py" in step
    assert "tests/test_model_nagumo_sato_map_neuron.py" in step
    assert "tests/test_nagumo_sato_map_backends.py" in step
    assert "tests/test_model_sc_adaptive_threshold_map_neuron.py" in step
    assert "tests/test_sc_adaptive_threshold_map_backends.py" in step
    assert "tests/test_model_kilinc_bhatt_map_neuron.py" in step
    assert "tests/test_kilinc_bhatt_map_backends.py" in step
    assert "--fail-under=100 --show-missing" in step


def test_ih_exact_coverage_and_backend_parity_is_hosted() -> None:
    """The omitted channel model needs an explicit branch gate and backend custody."""
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("- name: Ih exact coverage and backend parity")
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.coverage && matrix.shard-index == 3" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "--include='src/sc_neurocore/neurons/models/ih_neuron.py'" in step
    assert "tests/test_model_ih_neuron.py" in step
    assert "tests/test_ih_neuron_backends.py" in step
    assert "--fail-under=100 --show-missing" in step


def test_persistent_na_exact_coverage_and_backend_parity_is_hosted() -> None:
    """The omitted INaP model needs an explicit branch gate and backend custody."""
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("- name: PersistentNa exact coverage and backend parity")
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.coverage && matrix.shard-index == 3" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "--include='src/sc_neurocore/neurons/models/persistent_na_neuron.py'" in step
    assert "tests/test_model_persistent_na_neuron.py" in step
    assert "tests/test_persistent_na_neuron_backends.py" in step
    assert "--fail-under=100 --show-missing" in step


def test_nmda_exact_coverage_and_backend_parity_is_hosted() -> None:
    """The omitted NMDA channel model needs an explicit branch gate and backend custody."""
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("- name: NMDA exact coverage and backend parity")
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.coverage && matrix.shard-index == 3" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "--include='src/sc_neurocore/neurons/models/nmda_neuron.py'" in step
    assert "tests/test_model_nmda_neuron.py" in step
    assert "tests/test_nmda_neuron_backends.py" in step
    assert "--fail-under=100 --show-missing" in step


def test_sk_exact_coverage_and_backend_parity_is_hosted() -> None:
    """The omitted SK channel model needs an explicit branch gate and backend custody."""
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("- name: SK exact coverage and backend parity")
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.coverage && matrix.shard-index == 3" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "--include='src/sc_neurocore/neurons/models/sk_neuron.py'" in step
    assert "tests/test_model_sk_neuron.py" in step
    assert "tests/test_sk_neuron_backends.py" in step
    assert "--fail-under=100 --show-missing" in step


def test_ttype_ca_exact_coverage_and_backend_parity_is_hosted() -> None:
    """The omitted T-type model needs an explicit branch gate and backend custody."""
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("- name: TTypeCa exact coverage and backend parity")
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.coverage && matrix.shard-index == 3" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "--include='src/sc_neurocore/neurons/models/ttype_ca_neuron.py'" in step
    assert "tests/test_model_ttype_ca_neuron.py" in step
    assert "tests/test_ttype_ca_neuron_backends.py" in step
    assert "--fail-under=100 --show-missing" in step


def test_glm_exact_coverage_and_backend_parity_is_hosted() -> None:
    """The omitted GLM model needs an explicit branch gate and backend custody."""
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("- name: GLM exact coverage and backend parity")
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.coverage && matrix.shard-index == 4" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "--include='src/sc_neurocore/neurons/models/glm_neuron.py'" in step
    assert "tests/test_model_glm_neuron_glm_atomicity.py" in step
    assert "tests/test_glm_neuron_backends.py" in step
    assert "--fail-under=100 --show-missing" in step


def test_mainen_sejnowski_exact_coverage_and_backend_parity_is_hosted() -> None:
    """The omitted two-compartment model needs a branch gate and backend custody."""
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("- name: MainenSejnowski exact coverage and backend parity")
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.coverage && matrix.shard-index == 4" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "--include='src/sc_neurocore/neurons/models/mainen_sejnowski.py'" in step
    assert "tests/test_model_mainen_sejnowski_ms_atomicity.py" in step
    assert "tests/test_mainen_sejnowski_backends.py" in step
    assert "--fail-under=100 --show-missing" in step


def test_tc_lif_exact_coverage_and_backend_parity_is_hosted() -> None:
    """The TC-LIF family (canonical + both SC identities) needs branch custody."""
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("- name: TwoCompartmentLIF exact coverage and backend parity")
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.coverage && matrix.shard-index == 4" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "src/sc_neurocore/neurons/models/tc_lif.py" in step
    assert "src/sc_neurocore/neurons/models/sc_leaky_tc_lif.py" in step
    assert "src/sc_neurocore/neurons/models/sc_exponential_tc_lif.py" in step
    assert "tests/test_model_tc_lif_atomicity.py" in step
    assert "tests/test_tc_lif_backends.py" in step
    assert "--fail-under=100 --show-missing" in step


def test_psn_exact_coverage_and_backend_parity_is_hosted() -> None:
    """The sliding PSN family (canonical + SC identity) needs branch custody."""
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("- name: ParallelSpiking exact coverage and backend parity")
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.coverage && matrix.shard-index == 4" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "src/sc_neurocore/neurons/models/psn.py" in step
    assert "src/sc_neurocore/neurons/models/sc_resetting_psn.py" in step
    assert "tests/test_model_psn_atomicity.py" in step
    assert "tests/test_psn_backends.py" in step
    assert "--fail-under=100 --show-missing" in step


def test_hdc_exact_coverage_is_hosted() -> None:
    """The rebuilt HDC surface needs an explicit branch gate."""
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("- name: HDC exact coverage")
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.coverage && matrix.shard-index == 4" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "src/sc_neurocore/hdc/base.py" in step
    assert "src/sc_neurocore/hdc/classifier.py" in step
    assert "tests/test_hdc/" in step
    assert "--fail-under=100 --show-missing" in step


def test_exactly_one_primary_matrix_leg_carries_native_desktop_admission() -> None:
    """Require exactly one primary interpreter in the test matrix.

    Native desktop and browser admission run once. Other focused gates are
    distributed among the four coverage shards, without changing their commands.
    """
    try:
        import yaml
    except ModuleNotFoundError:  # pragma: no cover - PyYAML ships with the dev env.
        import pytest

        pytest.skip("PyYAML is not installed")
    workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    include = workflow["jobs"]["test"]["strategy"]["matrix"]["include"]
    primary_legs = [leg for leg in include if leg.get("primary") is True]
    assert len(primary_legs) == 1, (
        "the test matrix must declare exactly one primary leg; "
        f"found {len(primary_legs)} of {len(include)}"
    )
    assert primary_legs[0] == {"python-minor": "3.12", "shard-index": 1, "primary": True}
