# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def _repo_root() -> Path:
    """Return the repository root for release workflow contract checks."""
    return Path(__file__).resolve().parents[1]


def _workflow() -> dict[str, Any]:
    """Load the live publish workflow as a mapping."""
    path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "publish.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)
    return cast(dict[str, Any], workflow)


def _run_text(job: dict[str, Any]) -> str:
    """Return the concatenated shell commands from a workflow job."""
    steps = job["steps"]
    assert isinstance(steps, list)
    return "\n".join(step["run"] for step in steps if isinstance(step, dict) and "run" in step)


def _uses_step(job: dict[str, Any], action_prefix: str) -> dict[str, Any]:
    steps = job["steps"]
    assert isinstance(steps, list)
    matches = [
        step
        for step in steps
        if isinstance(step, dict) and str(step.get("uses", "")).startswith(action_prefix)
    ]
    assert len(matches) == 1
    return cast(dict[str, Any], matches[0])


def test_python_and_engine_pypi_publish_use_oidc_trusted_publishing() -> None:
    jobs = _workflow()["jobs"]

    python_publish = jobs["publish-python-pypi"]
    assert python_publish["environment"] == "pypi"
    assert python_publish["permissions"] == {"id-token": "write"}

    engine_publish = jobs["publish-engine-pypi"]
    assert engine_publish["environment"] == "pypi"
    assert engine_publish["permissions"] == {"id-token": "write"}

    engine_publish_text = str(engine_publish)
    assert "PYPI_ENGINE_TOKEN" in engine_publish_text
    assert "password" in engine_publish_text
    assert "attestations: false" not in engine_publish_text


def test_pypi_publish_steps_are_idempotent_for_tag_retries() -> None:
    jobs = _workflow()["jobs"]

    for job_name in ("publish-python-pypi", "publish-engine-pypi"):
        publish_step = _uses_step(jobs[job_name], "pypa/gh-action-pypi-publish@")
        assert publish_step["with"]["skip-existing"] is True


def test_crates_publish_skips_existing_engine_version_before_upload() -> None:
    jobs = _workflow()["jobs"]
    publish_text = _run_text(jobs["publish-crate"])

    assert "CRATE_NAME=" in publish_text
    assert "CRATE_VERSION=" in publish_text
    assert "https://crates.io/api/v1/crates/${CRATE_NAME}/${CRATE_VERSION}" in publish_text
    assert "HTTP_STATUS" in publish_text
    assert '"200")' in publish_text
    assert "already exists on crates.io" in publish_text
    assert "cargo publish --manifest-path engine/Cargo.toml" in publish_text


def test_publish_workflow_builds_and_smoke_tests_engine_wheels_before_upload() -> None:
    jobs = _workflow()["jobs"]

    build_text = _run_text(jobs["build-engine-wheels"])
    assert "maturin build --release --out ../dist-engine/" in build_text

    smoke = jobs["smoke-engine-wheels"]
    assert smoke["needs"] == ["build-engine-wheels", "build-engine-linux-arm"]
    smoke_text = _run_text(smoke)
    assert "pip install dist-engine/*.whl" in smoke_text
    assert "ZipFile(wheels[0])" not in smoke_text
    assert "PYTHONPATH=dist-engine/smoke" not in smoke_text
    assert "pip install --upgrade pip" not in smoke_text
    assert "import sc_neurocore_engine" in smoke_text
    assert "simd_tier()" in smoke_text

    assert jobs["publish-engine-pypi"]["needs"] == ["smoke-engine-wheels"]


def test_publish_workflow_engine_wheel_release_matrix_matches_ci() -> None:
    """Release builds the same prebuilt engine wheel targets as wheel CI."""
    jobs = _workflow()["jobs"]
    build_matrix = jobs["build-engine-wheels"]["strategy"]["matrix"]
    arm_matrix = jobs["build-engine-linux-arm"]["strategy"]["matrix"]

    assert build_matrix["os"] == ["ubuntu-latest", "windows-latest", "macos-latest"]
    assert build_matrix["python-version"] == ["3.10", "3.11", "3.12", "3.13", "3.14"]
    assert arm_matrix["python-version"] == build_matrix["python-version"]

    build_text = _run_text(jobs["build-engine-wheels"])
    arm_text = _run_text(jobs["build-engine-linux-arm"])
    assert "Remove tracked Windows bridge binary" in str(jobs["build-engine-wheels"])
    assert "maturin build --release --out ../dist-engine/" in build_text
    assert "aarch64-unknown-linux-gnu" in arm_text
    assert "--interpreter python${{ matrix.python-version }}" in arm_text


def test_rust_engine_release_fallback_docs_name_wheel_and_source_paths() -> None:
    """Docs keep the prebuilt wheel path and local Rust fallback discoverable."""
    root = _repo_root()
    docs = "\n".join(
        (
            (root / "README.md").read_text(encoding="utf-8"),
            (root / "docs" / "guides" / "install_profiles.md").read_text(encoding="utf-8"),
            (root / "docs" / "studio" / "index.md").read_text(encoding="utf-8"),
            (root / "docs" / "development" / "v3_migration.md").read_text(encoding="utf-8"),
        )
    )

    assert "python -m pip install sc_neurocore_engine" in docs
    assert "python -m maturin develop --release" in docs
    assert "Python 3.10-3.14" in docs
    assert "Linux x86_64/aarch64, macOS, and Windows" in docs
    assert "local Rust toolchain" in docs
