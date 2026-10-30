# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Release-workflow contracts: trusted publishing, idempotent uploads and the wheel matrix."""

from __future__ import annotations

import hashlib
import os
import subprocess
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
    return "\n".join(
        step["run"] for step in steps if isinstance(step, dict) and "run" in step
    )


def _uses_step(job: dict[str, Any], action_prefix: str) -> dict[str, Any]:
    """Return the single step of ``job`` whose ``uses`` starts with ``action_prefix``."""
    steps = job["steps"]
    assert isinstance(steps, list)
    matches = [
        step
        for step in steps
        if isinstance(step, dict)
        and str(step.get("uses", "")).startswith(action_prefix)
    ]
    assert len(matches) == 1
    return cast(dict[str, Any], matches[0])


def test_python_and_engine_pypi_publish_use_oidc_trusted_publishing() -> None:
    """Both PyPI publish jobs run in the pypi environment with an OIDC token permission."""
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
    """A retried tag run skips files PyPI already holds instead of failing the upload."""
    jobs = _workflow()["jobs"]

    for job_name in ("publish-python-pypi", "publish-engine-pypi"):
        publish_step = _uses_step(jobs[job_name], "pypa/gh-action-pypi-publish@")
        assert publish_step["with"]["skip-existing"] is True


def test_crates_publish_skips_existing_engine_version_before_upload() -> None:
    """The crate job checks crates.io for the engine version before it publishes."""
    jobs = _workflow()["jobs"]
    publish_text = _run_text(jobs["publish-crate"])

    assert "CRATE_NAME=" in publish_text
    assert "CRATE_VERSION=" in publish_text
    assert (
        "https://crates.io/api/v1/crates/${CRATE_NAME}/${CRATE_VERSION}" in publish_text
    )
    assert "HTTP_STATUS" in publish_text
    assert '"200")' in publish_text
    assert "already exists on crates.io" in publish_text
    assert "cargo publish --manifest-path engine/Cargo.toml" in publish_text


def test_publish_workflow_builds_and_smoke_tests_engine_wheels_before_upload() -> None:
    """Engine wheels are built and installed in a smoke test before the PyPI upload."""
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
    assert build_matrix["python-version"] == ["3.11", "3.12", "3.13", "3.14"]
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
            (root / "docs" / "guides" / "install_profiles.md").read_text(
                encoding="utf-8"
            ),
            (root / "docs" / "studio" / "index.md").read_text(encoding="utf-8"),
            (root / "docs" / "development" / "v3_migration.md").read_text(
                encoding="utf-8"
            ),
        )
    )

    assert "python -m pip install sc_neurocore_engine" in docs
    assert "python -m maturin develop --release" in docs
    assert "Python 3.11-3.14" in docs
    assert "Linux x86_64/aarch64, macOS, and Windows" in docs
    assert "local Rust toolchain" in docs


def test_python_dist_is_accepted_installed_before_it_can_be_published() -> None:
    """The wheel PyPI receives serves its Studio from a clean installation."""
    jobs = _workflow()["jobs"]
    build = _run_text(jobs["build-python-dist"])
    steps = [
        step for step in jobs["build-python-dist"]["steps"] if isinstance(step, dict)
    ]
    names = [step.get("name", step.get("run", step.get("uses", ""))) for step in steps]

    assert names.index("Accept the installed Studio the wheel provides") > names.index(
        "python -m build"
    )
    assert "--require-hashes" in build
    assert "--no-deps dist/sc_neurocore-*.whl" in build
    assert (
        'tools/studio_installed_acceptance.py --python "$RUNNER_TEMP/installed/bin/python"'
        in (build)
    )
    assert jobs["publish-python-pypi"]["needs"] == ["build-python-dist"]
    assert (_repo_root() / "tools" / "studio_installed_acceptance.py").is_file()


def _named_workflow(name: str) -> dict[str, Any]:
    """Load one workflow of the checkout by file name.

    Parameters
    ----------
    name : str
        File name under ``.github/workflows``.

    Returns
    -------
    dict
        The parsed workflow.
    """
    workflow = yaml.safe_load(
        (_repo_root() / ".github" / "workflows" / name).read_text("utf-8")
    )
    assert isinstance(workflow, dict)
    return cast(dict[str, Any], workflow)


def _step(job: dict[str, Any], name: str) -> tuple[int, dict[str, Any]]:
    """Return the position and body of the single step called ``name``.

    Parameters
    ----------
    job : dict
        Workflow job.
    name : str
        Step name.

    Returns
    -------
    tuple of int and dict
        Index of the step and the step.
    """
    found = [
        (i, s)
        for i, s in enumerate(job["steps"])
        if isinstance(s, dict) and s.get("name") == name
    ]
    assert len(found) == 1, name
    return found[0]


def _run_release_notes_step(
    tmp_path: Path, notes: str
) -> subprocess.CompletedProcess[str]:
    """Run the release-notes step of ``release.yml`` against written notes and dummy artefacts.

    Parameters
    ----------
    tmp_path : Path
        Working directory that stands in for the checkout.
    notes : str
        Text of ``docs/development/RELEASE_NOTES_v4.0.0.md`` in that checkout.

    Returns
    -------
    subprocess.CompletedProcess
        The finished step, with ``GITHUB_OUTPUT`` written to ``tmp_path/github_output``.
    """
    release = _named_workflow("release.yml")["jobs"]["release"]
    script = _step(release, "Release notes for this tag")[1]["run"]
    (tmp_path / "docs/development").mkdir(parents=True)
    (tmp_path / "docs/development/RELEASE_NOTES_v4.0.0.md").write_text(notes, "utf-8")
    (tmp_path / "dist").mkdir()
    for name in (
        "dist/sc_neurocore-4.0.0.tar.gz",
        "dist/sc_neurocore-4.0.0-py3-none-any.whl",
    ):
        (tmp_path / name).write_bytes(b"artefact " + name.encode())
    (tmp_path / "sbom.json").write_text("{}", "utf-8")
    (tmp_path / "sbom-engine-rust.json").write_text("{}", "utf-8")
    return subprocess.run(
        [
            "bash",
            "-e",
            "-c",
            script.replace("/tmp/release_notes.md", str(tmp_path / "body.md")),
        ],
        cwd=tmp_path,
        env={
            "PATH": os.environ["PATH"],
            "RELEASE_TAG": "v4.0.0",
            "GITHUB_OUTPUT": str(tmp_path / "github_output"),
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_release_body_is_the_curated_notes_with_checksums_and_fits_github(
    tmp_path: Path,
) -> None:
    """The GitHub Release body is the version's curated notes plus every attachment's SHA-256.

    GitHub refuses bodies over 125,000 characters, and the full ``[4.0.0]`` CHANGELOG section is
    several times larger; a refused body would fail the release after the registries published.

    Parameters
    ----------
    tmp_path : Path
        Working directory for the step run.
    """
    notes = (_repo_root() / "docs/development/RELEASE_NOTES_v4.0.0.md").read_text(
        "utf-8"
    )
    assert len(notes.encode()) < 100_000, "leave room for the checksum table"
    assert "Studio 1.0.0" not in notes, (
        "4.0.0 ships SNN Studio as a development preview"
    )
    done = _run_release_notes_step(tmp_path, notes)
    assert done.returncode == 0, done.stderr
    body = (tmp_path / "body.md").read_text("utf-8")
    assert body.startswith(notes)
    for name in (
        "sc_neurocore-4.0.0.tar.gz",
        "sc_neurocore-4.0.0-py3-none-any.whl",
        "sbom-engine-rust.json",
    ):
        digest = hashlib.sha256((next(tmp_path.rglob(name))).read_bytes()).hexdigest()
        assert f"| `{name}` | `{digest}` |" in body
    assert (tmp_path / "github_output").read_text(
        "utf-8"
    ) == f"notes_file={tmp_path / 'body.md'}\n"


def test_release_body_refuses_missing_or_oversized_notes(tmp_path: Path) -> None:
    """The step fails closed instead of handing GitHub a body it would refuse.

    Parameters
    ----------
    tmp_path : Path
        Working directory for the step runs.
    """
    oversized = _run_release_notes_step(tmp_path / "big", "x" * 125_001)
    assert oversized.returncode != 0 and "GitHub allows 125000" in oversized.stderr
    empty = _run_release_notes_step(tmp_path / "empty", "")
    assert empty.returncode != 0 and "missing curated release notes" in empty.stderr


def test_release_sboms_describe_what_users_install() -> None:
    """The attached SBOMs cover the accepted install environment and the locked Rust graph."""
    job = _named_workflow("release.yml")["jobs"]["release"]
    sbom_position, sbom = _step(job, "Generate SBOM")
    assert (
        sbom_position > _step(job, "Accept the installed Studio the wheel provides")[0]
    )
    assert 'cyclonedx-py environment "$RUNNER_TEMP/installed/bin/python"' in sbom["run"]
    assert (
        "syft scan file:Cargo.lock -o cyclonedx-json=sbom-engine-rust.json"
        in sbom["run"]
    )
    published = [
        s
        for s in job["steps"]
        if "softprops/action-gh-release@" in str(s.get("uses", ""))
    ]
    assert "sbom-engine-rust.json" in published[0]["with"]["files"]


def test_unpublished_dispatch_runs_the_full_crate_verify_build() -> None:
    """``publish=false`` still proves the crate, while the upload job stays skipped."""
    jobs = _workflow()["jobs"]
    dry_run = jobs["dry-run-crate"]
    assert dry_run["if"] == (
        "github.event_name == 'workflow_dispatch' && github.event.inputs.publish != 'true'"
    )
    commands = _run_text(dry_run)
    assert (
        "cargo publish --manifest-path engine/Cargo.toml --dry-run --locked" in commands
    )
    assert "--no-verify" not in commands
    assert "--dry-run" not in _run_text(jobs["publish-crate"])


def test_container_image_is_scanned_before_any_push() -> None:
    """No image tag reaches GHCR before the CRITICAL/HIGH gate passed."""
    job = _named_workflow("docker.yml")["jobs"]["build-push"]
    build_at, build = _step(job, "Build the image locally")
    scan_at, scan = _step(job, "Trivy fail on CRITICAL/HIGH")
    push_at, push = _step(job, "Push the scanned image")
    assert build_at < scan_at < push_at
    assert build["with"]["load"] is True and build["with"]["push"] is False
    assert "--exit-code 1" in scan["run"]
    assert push["with"]["push"] is True
    pushes = [
        s
        for s in job["steps"]
        if isinstance(s, dict) and s.get("with", {}).get("push") is True
    ]
    assert pushes == [push]
    assert _step(job, "Extract metadata")[1]["with"]["flavor"].split() == [
        "latest=auto"
    ]
