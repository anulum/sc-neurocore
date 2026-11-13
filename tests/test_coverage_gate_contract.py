# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python coverage threshold cross-surface contract

"""Keep local, preflight, and hosted Python coverage gates at exact closure."""

from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Any

import tomllib

import pytest


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
    assert native["runs-on"] == "windows-2025-vs2026"
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
    assert _cov_thresholds(_ROOT / ".github/workflows/ci.yml") == (100, 100, 100)


def test_qualified_nir_inference_closes_all_new_interoperability_modules() -> None:
    """Measure installed framework code without inheriting the global NIR omission."""
    import yaml

    workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["test-nir-interoperability"]
    step = next(
        item
        for item in job["steps"]
        if item.get("name") == "Installed snnTorch graph identity and both inference directions"
    )
    assert step["if"] == "matrix.profile == 'snntorch'"
    commands = step["run"]
    assert "coverage run --rcfile=/dev/null --branch --omit=" in commands
    assert "coverage report --rcfile=/dev/null --omit= --fail-under=100" in commands
    for module in ("snntorch_bridge", "graph_identity", "execution_metadata"):
        assert f"*/nir_bridge/{module}.py" in commands


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
        "RECORDER_RESULT": "${{ needs.recorder-coverage.result }}",
    }
    assert readiness["run"] == (
        'test "$TEST_RESULT" = success && test "$WINDOWS_RESULT" = success '
        '&& test "$RECORDER_RESULT" = success'
    )
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
    timing_steps = jobs["test-timings"]["steps"]
    install = next(
        index
        for index, step in enumerate(timing_steps)
        if step.get("run") == "python -m pip install --require-hashes -r requirements/runtime.txt"
    )
    record = next(
        index
        for index, step in enumerate(timing_steps)
        if "tools/pytest_shards.py record" in step.get("run", "")
    )
    assert install < record, "the report parser needs the hash-locked defusedxml first"


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


def test_recorder_branch_floor_is_a_required_primary_dependency() -> None:
    """Keep native recorder inputs and the unchanged production floor mandatory."""
    import yaml

    workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text())
    recorder = workflow["jobs"]["recorder-coverage"]
    assert recorder["needs"] == ["test", "shd-non-linux-refusal"]
    assert recorder["if"] == "${{ !cancelled() }}"
    assert recorder["uses"] == "./.github/workflows/recorder-coverage.yml"
    assert "recorder-coverage" in workflow["jobs"]["coverage-union"]["needs"]
    reusable = yaml.safe_load((_ROOT / recorder["uses"]).read_text())
    job = reusable["jobs"]["recorder-coverage"]
    assert job["env"]["RECORDER_COVERAGE_FLOOR"] == "98.417721518987"
    downloads = [
        step["with"]["name"]
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/download-artifact@")
    ]
    assert downloads == ["facet-recorder-linux-coverage", "facet-recorder-windows-coverage"]
    floor = next(
        step
        for step in job["steps"]
        if step.get("name") == "Enforce the production recorder branch floor"
    )
    assert floor["run"].count('--fail-under="$RECORDER_COVERAGE_FLOOR"') == 2
    assert "continue-on-error" not in recorder
    assert "continue-on-error" not in floor


_RECORDER_SOURCES = (
    "tools/facet_receipt.py",
    "tools/facet_receipt_evidence.py",
    "tools/facet_receipt_runtime.py",
)
_RECORDER_MODULES = frozenset(
    path.removesuffix(".py").replace("/", ".") for path in _RECORDER_SOURCES
)


def _windows_side(test: ast.expr) -> str | None:
    """Name the side of an ``if`` that only Windows takes, from its platform test.

    Parameters
    ----------
    test : ast.expr
        The ``if`` statement's condition.

    Returns
    -------
    str | None
        ``"body"`` for ``sys.platform == "win32"``; ``"orelse"`` for
        ``sys.platform != "win32"``, alone or as an ``and`` operand; ``None``
        when the condition does not read ``sys.platform``.
    """
    operands = (
        test.values if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) else [test]
    )
    for operand in operands:
        if (
            isinstance(operand, ast.Compare)
            and ast.unparse(operand.left) == "sys.platform"
            and len(operand.ops) == 1
            and ast.unparse(operand.comparators[0]) == "'win32'"
        ):
            if isinstance(operand.ops[0], ast.NotEq):
                return "orelse"
            if isinstance(operand.ops[0], ast.Eq) and operand is test:
                return "body"
    assert "sys.platform" not in ast.unparse(test), (
        f"unsupported platform test at line {test.lineno}: {ast.unparse(test)}"
    )
    return None


def _windows_only_code(source: str) -> tuple[set[int], set[tuple[int, int]]]:
    """Derive the statements and branch arcs that only a Windows run executes.

    The Windows side of a platform test (see `_windows_side`) runs on Windows
    alone. Every statement inside it is Windows-only, and so is every arc out of
    an ``if`` inside it. The arc from the platform test into that side is
    Windows-only too; a test without an ``else`` hands Windows to the statement
    after it. A construct whose arcs this does not model is refused by name.

    Parameters
    ----------
    source : str
        One recorder module's text.

    Returns
    -------
    tuple[set[int], set[tuple[int, int]]]
        The Windows-only statement lines and branch arcs, numbered as coverage
        numbers them.
    """
    lines: set[int] = set()
    arcs: set[tuple[int, int]] = set()

    def false_side(statement: ast.If, after: int | None) -> int:
        if statement.orelse:
            return statement.orelse[0].lineno
        assert after is not None, f"no statement follows line {statement.lineno} in its block"
        return after

    def windows_only(block: list[ast.stmt], follower: int | None) -> None:
        for index, statement in enumerate(block):
            after = block[index + 1].lineno if index + 1 < len(block) else follower
            lines.add(statement.lineno)
            if isinstance(statement, ast.If):
                arcs.add((statement.lineno, statement.body[0].lineno))
                arcs.add((statement.lineno, false_side(statement, after)))
                windows_only(statement.body, after)
                windows_only(statement.orelse, after)
            elif isinstance(statement, ast.With):
                windows_only(statement.body, None)
            else:
                assert not any(
                    isinstance(child, ast.stmt) for child in ast.iter_child_nodes(statement)
                ), f"unmodelled Windows-only construct at line {statement.lineno}"

    def visit(block: list[ast.stmt], follower: int | None) -> None:
        for index, statement in enumerate(block):
            after = block[index + 1].lineno if index + 1 < len(block) else follower
            if isinstance(statement, ast.If):
                side = _windows_side(statement.test)
                if side == "body":
                    arcs.add((statement.lineno, statement.body[0].lineno))
                    windows_only(statement.body, after)
                    visit(statement.orelse, after)
                elif side == "orelse":
                    arcs.add((statement.lineno, false_side(statement, after)))
                    windows_only(statement.orelse, after)
                    visit(statement.body, after)
                else:
                    visit(statement.body, after)
                    visit(statement.orelse, after)
                continue
            for field in ("body", "orelse", "finalbody"):
                children = getattr(statement, field, None)
                if isinstance(children, list):
                    visit(children, None)
            for handler in getattr(statement, "handlers", []):
                visit(handler.body, None)

    visit(ast.parse(source).body, None)
    return lines, arcs


def _heredoc_program(run: str) -> ast.Module:
    """Parse the Python heredoc of one workflow step.

    Parameters
    ----------
    run : str
        The step's shell text.

    Returns
    -------
    ast.Module
        The program between ``python - <<'PY'`` and its closing ``PY`` line.
    """
    match = re.search(r"^python - <<'PY'\n(.*?)\nPY$", run, re.DOTALL | re.MULTILINE)
    assert match is not None, "the step must run one Python heredoc"
    return ast.parse(match.group(1))


def _recorder_allowances(
    job: dict[str, Any],
) -> tuple[set[int], dict[str, tuple[set[int], set[tuple[int, int]]]]]:
    """Read the Windows-only lines that the recorder job's two Python steps pin.

    Parameters
    ----------
    job : dict[str, Any]
        The reusable workflow's ``recorder-coverage`` job.

    Returns
    -------
    tuple[set[int], dict[str, tuple[set[int], set[tuple[int, int]]]]]
        The lines the proof step requires Windows to have run, and the floor
        step's allowed missing lines and branch arcs per recorder source.
    """
    steps = {step.get("name"): step for step in job["steps"]}
    proof = _heredoc_program(steps["Prove canonical paths and actual Windows contribution"]["run"])
    native = [
        node.value.right
        for node in ast.walk(proof)
        if isinstance(node, ast.Assign)
        and [ast.unparse(target) for target in node.targets] == ["native_lines"]
        and isinstance(node.value, ast.BinOp)
    ]
    assert len(native) == 1, "the proof step must pin its Windows-only lines once"
    floor = _heredoc_program(steps["Enforce the production recorder branch floor"]["run"])
    allowed = [
        node.value
        for node in ast.walk(floor)
        if isinstance(node, ast.AnnAssign) and ast.unparse(node.target) == "allowed"
    ]
    assert len(allowed) == 1 and isinstance(allowed[0], ast.Dict), (
        "the floor must pin one allowance table"
    )
    table: dict[str, tuple[set[int], set[tuple[int, int]]]] = {}
    for key, value in zip(allowed[0].keys, allowed[0].values, strict=True):
        assert key is not None and isinstance(value, ast.Tuple) and len(value.elts) == 2
        lines, arcs = (
            set() if ast.unparse(part) == "set()" else ast.literal_eval(part) for part in value.elts
        )
        table[ast.literal_eval(key)] = (lines, arcs)
    return ast.literal_eval(native[0]), table


def test_windows_only_derivation_reproduces_the_reviewed_allowances() -> None:
    """The derivation gives the pinned sets that were reviewed against the original layout.

    The recorder at ``de9cc74c7`` placed its Windows-only code at lines 75-76
    and 170-171; those numbers and six branches were the reviewed allowance.
    """
    source = """\
import sys
def read(descriptor, size):
    if size == 0:
        return b""
    if sys.platform == "win32":
        with open(descriptor) as captured:
            return captured
    return b""
if sys.platform != "win32":
    def wait():
        return None
def run(process, timed_out):
    try:
        code = process.wait()
        if sys.platform != "win32" and os.name == "posix":
            wait()
        captured = read(1, 2)
    except OSError:
        pass
    if timed_out:
        if sys.platform != "win32" and os.name == "posix":
            os.killpg(process.pid, 9)
        elif process.poll() is None:
            process.kill()
        code = process.wait()
    return code
"""
    lines, arcs = _windows_only_code(source)
    assert lines == {6, 7, 23, 24}
    assert arcs == {(5, 6), (9, 12), (15, 17), (21, 23), (23, 24), (23, 25)}


def test_windows_only_derivation_follows_both_sides_of_a_nested_if() -> None:
    """Both arcs out of an ``if`` inside a Windows-only block are Windows-only."""
    source = """\
import sys
if sys.platform == "win32":
    if flag:
        value = 1
    else:
        value = 2
else:
    value = 3
done = value
"""
    lines, arcs = _windows_only_code(source)
    assert lines == {3, 4, 6}
    assert arcs == {(2, 3), (3, 4), (3, 6)}


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            'import sys\nif sys.platform == "win32" and flag:\n    pass\n',
            "unsupported platform test at line 2",
        ),
        (
            'import sys\nif sys.platform == "win32":\n    for item in items:\n        pass\n',
            "unmodelled",
        ),
        ('import sys\nif sys.platform != "win32":\n    pass\n', "no statement follows line 2"),
    ],
)
def test_windows_only_derivation_refuses_what_it_does_not_model(source: str, message: str) -> None:
    """An unmodelled platform shape fails by name instead of yielding a wrong set.

    Parameters
    ----------
    source : str
        A module with one unmodelled shape.
    message : str
        The text the refusal must carry.
    """
    with pytest.raises(AssertionError, match=message):
        _windows_only_code(source)


def test_recorder_windows_allowances_follow_the_recorder_source() -> None:
    """The recorder job's pinned Windows-only numbers move with the source.

    Both Python steps of ``recorder-coverage.yml`` pin coverage numbers: the
    proof requires Windows to have run one of the Windows-only lines, and the
    floor allows exactly those lines and branches to stay uncovered. A source
    edit above them shifts the numbers, which the job only notices after the
    push. Each pinned set must equal the one derived from the current source.
    """
    import yaml

    reusable = yaml.safe_load(
        (_ROOT / ".github/workflows/recorder-coverage.yml").read_text(encoding="utf-8")
    )
    native, table = _recorder_allowances(reusable["jobs"]["recorder-coverage"])
    derived = {
        path: _windows_only_code((_ROOT / path).read_text(encoding="utf-8"))
        for path in _RECORDER_SOURCES
    }
    assert set(table) == set(_RECORDER_SOURCES)
    for path in _RECORDER_SOURCES:
        assert table[path] == derived[path], (
            f"{path}: the floor pins {table[path]}, the source derives {derived[path]}"
        )
    assert native == derived["tools/facet_receipt_evidence.py"][0], (
        f"the proof pins {native}, the source derives {derived['tools/facet_receipt_evidence.py'][0]}"
    )


def _imports_a_recorder_tool(path: Path) -> bool:
    """Report whether a test module imports one of the recorder tools.

    Parameters
    ----------
    path : Path
        A test module.

    Returns
    -------
    bool
        True when an ``import`` anywhere in the module names a recorder module.
    """
    text = path.read_text(encoding="utf-8")
    if "facet_receipt" not in text:
        return False
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import) and any(
            alias.name in _RECORDER_MODULES for alias in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module in _RECORDER_MODULES:
                return True
            if any(f"{node.module}.{alias.name}" in _RECORDER_MODULES for alias in node.names):
                return True
    return False


@pytest.mark.parametrize(
    ("text", "imports"),
    [
        ("import tools.facet_receipt_evidence\n", True),
        ("from tools.facet_receipt import run_command\n", True),
        ("from tools import facet_receipt_runtime\n", True),
        ("def check():\n    from tools.facet_receipt_evidence import capture_execution\n", True),
        ("from tools import helper\nfrom . import facet_receipt\n", False),
        ("NAME = 'tools/facet_receipt.py'\n", False),
        ("import json\n", False),
    ],
)
def test_recorder_import_reader_names_each_import_form(
    tmp_path: Path, text: str, imports: bool
) -> None:
    """Every import form counts; a path in data or a relative import does not.

    Parameters
    ----------
    tmp_path : Path
        Folder for the probed module.
    text : str
        The module's text.
    imports : bool
        Whether the module imports a recorder tool.
    """
    module = tmp_path / "test_probe.py"
    module.write_text(text, encoding="utf-8")
    assert _imports_a_recorder_tool(module) is imports


def test_every_recorder_importing_test_feeds_the_linux_recorder_data() -> None:
    """A recorder test outside the primary producer leaves its lines to the floor.

    The recorder floor reads the primary leg's listed tests and the Windows
    reaping test only. A test module that imports a recorder tool but is not
    listed covers its lines in the ordinary shards alone; the floor then
    refuses those lines as uncovered portable statements after the push.
    """
    import yaml

    workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    producer = [
        step
        for step in workflow["jobs"]["test"]["steps"]
        if step.get("name") == "Recorder execution and publication branch coverage"
    ]
    assert len(producer) == 1
    listed = set(re.findall(r"\btests/\S+\.py\b", producer[0]["run"]))
    importing = {
        path.relative_to(_ROOT).as_posix()
        for path in (_ROOT / "tests").rglob("test_*.py")
        if _imports_a_recorder_tool(path)
    }
    assert "tests/test_facet_receipt_process.py" in importing
    assert importing <= listed, (
        f"recorder tests the producer does not run: {sorted(importing - listed)}"
    )


@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped"])
def test_recorder_result_controls_the_actual_primary_union_shell(result: str) -> None:
    """Execute the real union guard for each recorder terminal result.

    Parameters
    ----------
    result : str
        Recorder job result; only success may admit the primary union.
    """
    import os
    import subprocess

    import yaml

    workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text())
    guard = workflow["jobs"]["coverage-union"]["steps"][0]
    environment = dict(os.environ)
    environment.update(TEST_RESULT="success", WINDOWS_RESULT="success", RECORDER_RESULT=result)
    completed = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", guard["run"]],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert (completed.returncode == 0) == (result == "success")
