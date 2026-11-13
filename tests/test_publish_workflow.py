# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Release workflow contracts

"""Release-workflow contracts: trusted publishing, idempotent uploads and the wheel matrix."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
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
    """Return the single step of ``job`` whose ``uses`` starts with ``action_prefix``."""
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
    """Both PyPI publish jobs run in the pypi environment with an OIDC token permission."""
    jobs = _workflow()["jobs"]

    python_publish = jobs["publish-python-pypi"]
    assert python_publish["environment"] == "pypi"
    assert python_publish["permissions"] == {"id-token": "write"}

    engine_publish = jobs["publish-engine-pypi"]
    assert engine_publish["environment"] == "pypi"
    assert engine_publish["permissions"] == {"id-token": "write"}

    # Neither upload carries an API token, so PyPI authenticates the job's OIDC
    # identity and the action keeps its default provenance attestations.
    for job_name in ("publish-python-pypi", "publish-engine-pypi"):
        upload = _uses_step(jobs[job_name], "pypa/gh-action-pypi-publish@")["with"]
        assert "password" not in upload and "user" not in upload
        assert "attestations" not in upload
    workflow_text = (_repo_root() / ".github" / "workflows" / "publish.yml").read_text("utf-8")
    assert "secrets.PYPI" not in workflow_text
    security = " ".join((_repo_root() / "SECURITY.md").read_text("utf-8").split())
    assert "Both PyPI packages, `sc-neurocore` and `sc-neurocore-engine`" in security


def test_pypi_publish_steps_are_idempotent_for_tag_retries() -> None:
    """A retried tag run skips files PyPI already holds instead of failing the upload."""
    jobs = _workflow()["jobs"]

    for job_name in ("publish-python-pypi", "publish-engine-pypi"):
        publish_step = _uses_step(jobs[job_name], "pypa/gh-action-pypi-publish@")
        assert publish_step["with"]["skip-existing"] is True


_CRATE_STATUS = "python3 tools/crates_io_release_status.py engine/Cargo.toml"


def test_crates_publish_skips_existing_engine_version_before_upload() -> None:
    """The crate job asks the shared status tool and uploads only an unpublished version."""
    publish_text = _run_text(_workflow()["jobs"]["publish-crate"])

    assert f'STATUS="$({_CRATE_STATUS})"' in publish_text
    assert "published)" in publish_text and "unpublished)" in publish_text
    assert "already exists on crates.io; skipping publish." in publish_text
    assert publish_text.index("unpublished)") < publish_text.index(
        "cargo publish --manifest-path engine/Cargo.toml"
    )
    assert "Unexpected crates.io release status" in publish_text


def test_rollback_plan_completes_a_partial_publication_only_by_existing_mechanisms() -> None:
    """The plan's recovery of a partly published tag names only what the workflows provide.

    Re-running a tag run is safe only because both PyPI uploads skip existing files and the
    crate job skips a listed version; the GitHub Release backfill is a ``release.yml``
    dispatch with ``tag`` and ``publish`` inputs that checks out that tag.
    """
    plan = " ".join(
        (_repo_root() / "docs" / "development" / "release_rollback.md").read_text("utf-8").split()
    )
    section = plan[
        plan.index("## Partial publication") : plan.index("## Withdraw the faulty version")
    ]
    for workflow_name in ("publish.yml", "docker.yml", "release.yml"):
        assert f"`{workflow_name}`" in section
        events = cast(dict[Any, Any], _named_workflow(workflow_name)).get(True, {})
        assert events["push"]["tags"] == ["v*"]
    assert "`skip-existing`" in section
    jobs = _workflow()["jobs"]
    for job_name in ("publish-python-pypi", "publish-engine-pypi"):
        assert _uses_step(jobs[job_name], "pypa/gh-action-pypi-publish@")["with"]["skip-existing"]
    assert "skipping publish." in _run_text(jobs["publish-crate"])
    assert "dispatching `release.yml` with `tag`" in section and "`publish` set to true" in section
    release = _named_workflow("release.yml")
    inputs = cast(dict[Any, Any], release).get(True, {})["workflow_dispatch"]["inputs"]
    assert inputs["tag"]["required"] is True and inputs["publish"]["type"] == "boolean"
    checkout = release["jobs"]["release"]["steps"][0]
    assert "inputs.publish && inputs.tag" in checkout["with"]["ref"]
    # GHCR: once a tag is pushed, missing tags come from it through the alias job, not a rebuild.
    assert "`alias_from` and `alias_to` inputs of `docker.yml`" in section
    docker = _named_workflow("docker.yml")
    assert {"alias_from", "alias_to"} <= set(
        cast(dict[Any, Any], docker).get(True, {})["workflow_dispatch"]["inputs"]
    )
    alias_run = _run_text(docker["jobs"]["alias-image"])
    assert "docker buildx imagetools create" in alias_run and "build-push-action" not in str(
        docker["jobs"]["alias-image"]
    )


_UNTRUSTED_CONTEXT = re.compile(r"\$\{\{[^}]*\b(?:inputs\.|github\.event\.|github\.head_ref)")


def _script_interpolations(run: str) -> list[str]:
    """Return the lines of a step script that expand dispatch or event data as script text.

    Parameters
    ----------
    run : str
        Shell text of one workflow step.

    Returns
    -------
    list of str
        Lines in which an ``inputs``, ``github.event`` or ``github.head_ref`` expression is
        substituted into the script itself, where its value would be parsed as shell.
    """
    return [line.strip() for line in run.splitlines() if _UNTRUSTED_CONTEXT.search(line)]


def test_no_workflow_script_expands_dispatch_or_event_data_as_shell() -> None:
    """Dispatch inputs and event data reach every step script through ``env``, never as its text."""
    offenders = []
    for path in sorted((_repo_root() / ".github" / "workflows").glob("*.yml")):
        workflow = _named_workflow(path.name)
        for job_name, job in workflow["jobs"].items():
            for step in job.get("steps", []):
                for line in _script_interpolations(step.get("run", "")):
                    offenders.append(f"{path.name}:{job_name}: {line}")
    assert offenders == []


@pytest.mark.parametrize(
    ("run", "expanded"),
    [
        ('RELEASE_TAG="${{ inputs.tag }}"', True),
        ("-max_total_time=${{ github.event.inputs.max_total_time || '300' }}", True),
        ('echo "${{ github.head_ref }}"', True),
        ('RELEASE_TAG="${DISPATCH_TAG}"', False),
        ('if [ "${GITHUB_EVENT_NAME}" = "workflow_dispatch" ]; then', False),
        ("echo ${{ steps.meta.outputs.version }}", False),
    ],
)
def test_the_script_scan_recognises_an_expanded_input(run: str, expanded: bool) -> None:
    """The scan finds an input or event expression in script text and accepts env references.

    Parameters
    ----------
    run : str
        Step shell text.
    expanded : bool
        Whether the text substitutes dispatch or event data into the script.
    """
    assert bool(_script_interpolations(run)) is expanded


def test_both_crate_jobs_query_the_registry_through_one_tool() -> None:
    """A dispatch without publishing proves the same registry query the tag run needs."""
    jobs = _workflow()["jobs"]
    dry_run, publish = _run_text(jobs["dry-run-crate"]), _run_text(jobs["publish-crate"])
    assert _CRATE_STATUS in dry_run and _CRATE_STATUS in publish
    assert dry_run.index(_CRATE_STATUS) < dry_run.index("cargo publish")


def test_no_workflow_queries_the_crates_registry_by_hand() -> None:
    """Every crates.io API query goes through the tool that identifies the project."""
    for path in sorted((_repo_root() / ".github" / "workflows").glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        assert "crates.io/api" not in text, f"{path.name} queries crates.io outside the tool"


def test_publish_workflow_builds_and_smoke_tests_engine_wheels_before_upload() -> None:
    """Engine wheels are built and installed in a smoke test before the PyPI upload."""
    jobs = _workflow()["jobs"]

    build_text = _run_text(jobs["build-engine-wheels"])
    assert "maturin build --release --locked --out ../dist-engine/" in build_text

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


def test_every_uploaded_engine_wheel_is_smoke_tested_on_its_own_platform() -> None:
    """The smoke matrix installs exactly the artifacts the build jobs upload, each natively.

    Each build job names its artifact ``engine-wheel-<platform>-py<version>``; the smoke
    job downloads one such artifact per matrix cell, on a runner of that platform with that
    Python, so a wheel that cannot be installed or imported anywhere stops the upload.
    """
    jobs = _workflow()["jobs"]
    build = jobs["build-engine-wheels"]["strategy"]["matrix"]
    arm = jobs["build-engine-linux-arm"]["strategy"]["matrix"]
    built = {
        (os_name, version) for os_name in build["os"] for version in build["python-version"]
    } | {("linux-arm64", version) for version in arm["python-version"]}
    assert jobs["build-engine-wheels"]["steps"][-1]["with"]["name"] == (
        "engine-wheel-${{ matrix.os }}-py${{ matrix.python-version }}"
    )
    assert jobs["build-engine-linux-arm"]["steps"][-1]["with"]["name"] == (
        "engine-wheel-linux-arm64-py${{ matrix.python-version }}"
    )

    smoke = jobs["smoke-engine-wheels"]
    matrix = smoke["strategy"]["matrix"]
    runners = {"ubuntu-24.04": "ubuntu-24.04", "windows-2025-vs2026": "windows-2025-vs2026"}
    runners |= {"macos-26": "macos-26", "linux-arm64": "ubuntu-24.04-arm"}
    assert {target["artifact"]: target["runner"] for target in matrix["target"]} == runners
    smoked = {
        (target["artifact"], version)
        for target in matrix["target"]
        for version in matrix["python-version"]
    }
    assert smoked == built
    assert smoke["strategy"]["fail-fast"] is False
    assert smoke["runs-on"] == "${{ matrix.target.runner }}"
    setup = _uses_step(smoke, "actions/setup-python@")
    assert setup["with"]["python-version"] == "${{ matrix.python-version }}"
    download = _uses_step(smoke, "actions/download-artifact@")["with"]
    assert (
        download["name"]
        == "engine-wheel-${{ matrix.target.artifact }}-py${{ matrix.python-version }}"
    )
    assert "pattern" not in download
    install = next(
        step for step in smoke["steps"] if "pip install dist-engine/*.whl" in step.get("run", "")
    )
    assert install["shell"] == "bash"
    # The installed engine must report the project version, so a stale wheel cannot pass.
    checkout = _uses_step(smoke, "actions/checkout@")["with"]
    assert checkout["sparse-checkout"] == "pyproject.toml"
    run = install["run"]
    assert 'tomllib.loads(Path("pyproject.toml")' in run
    assert 'importlib.metadata.version("sc-neurocore-engine")' in run
    assert "set(found.values()) != {expected}" in run


def test_publish_workflow_engine_wheel_release_matrix_matches_ci() -> None:
    """Release builds the same prebuilt engine wheel targets as wheel CI."""
    jobs = _workflow()["jobs"]
    build_matrix = jobs["build-engine-wheels"]["strategy"]["matrix"]
    arm_matrix = jobs["build-engine-linux-arm"]["strategy"]["matrix"]

    assert build_matrix["os"] == ["ubuntu-24.04", "windows-2025-vs2026", "macos-26"]
    assert build_matrix["python-version"] == ["3.11", "3.12", "3.13", "3.14"]
    assert arm_matrix["python-version"] == build_matrix["python-version"]

    build_text = _run_text(jobs["build-engine-wheels"])
    arm_text = _run_text(jobs["build-engine-linux-arm"])
    assert "maturin build --release --locked --out ../dist-engine/" in build_text
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
    assert "Python 3.11-3.14" in docs
    assert "Linux x86_64/aarch64, macOS, and Windows" in docs
    assert "local Rust toolchain" in docs


def test_python_dist_is_accepted_installed_before_it_can_be_published() -> None:
    """The wheel PyPI receives serves its Studio from a clean installation."""
    jobs = _workflow()["jobs"]
    build = _run_text(jobs["build-python-dist"])
    steps = [step for step in jobs["build-python-dist"]["steps"] if isinstance(step, dict)]
    names = [step.get("name", step.get("run", step.get("uses", ""))) for step in steps]

    assert names.index("Accept the installed Studio the wheel provides") > names.index(
        "python -m build"
    )
    assert "--require-hashes" in build
    assert "--no-deps dist/sc_neurocore-*.whl" in build
    assert 'tools/studio_installed_acceptance.py --python "$RUNNER_TEMP/installed/bin/python"' in (
        build
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
    workflow = yaml.safe_load((_repo_root() / ".github" / "workflows" / name).read_text("utf-8"))
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
        (i, s) for i, s in enumerate(job["steps"]) if isinstance(s, dict) and s.get("name") == name
    ]
    assert len(found) == 1, name
    return found[0]


#: Every file the release attaches, as the workflow's job produces them.
RELEASE_ATTACHMENTS = (
    "dist/sc_neurocore-4.0.0.tar.gz",
    "dist/sc_neurocore-4.0.0-py3-none-any.whl",
    "sbom.json",
    "sbom-engine-rust.json",
    "security/ci-security-packet/release_security_artifact_index.json",
    "security/ci-security-packet/security/release_security_sweep_summary.json",
)


def _run_release_notes_step(
    tmp_path: Path, notes: str, *, omit: tuple[str, ...] = (), empty: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the release-notes step of ``release.yml`` against written notes and dummy artefacts.

    Parameters
    ----------
    tmp_path : Path
        Working directory that stands in for the checkout.
    notes : str
        Text of ``docs/development/RELEASE_NOTES_v4.0.0.md`` in that checkout.
    omit : tuple of str
        Entries of ``RELEASE_ATTACHMENTS`` that are not written.
    empty : str or None
        One entry of ``RELEASE_ATTACHMENTS`` that is written with no bytes.

    Returns
    -------
    subprocess.CompletedProcess
        The finished step, with ``GITHUB_OUTPUT`` written to ``tmp_path/github_output``.
    """
    release = _named_workflow("release.yml")["jobs"]["release"]
    script = _step(release, "Release notes for this tag")[1]["run"]
    (tmp_path / "docs/development").mkdir(parents=True)
    (tmp_path / "docs/development/RELEASE_NOTES_v4.0.0.md").write_text(notes, "utf-8")
    for name in RELEASE_ATTACHMENTS:
        if name in omit:
            continue
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"" if name == empty else b"artefact " + name.encode())
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
    notes = (_repo_root() / "docs/development/RELEASE_NOTES_v4.0.0.md").read_text("utf-8")
    assert len(notes.encode()) < 100_000, "leave room for the checksum table"
    assert "Studio 1.0.0" not in notes, "4.0.0 ships SNN Studio as a development preview"
    done = _run_release_notes_step(tmp_path, notes)
    assert done.returncode == 0, done.stderr
    body = (tmp_path / "body.md").read_text("utf-8")
    assert body.startswith(notes)
    rows = [line for line in body.splitlines() if line.startswith("| `")]
    assert len(rows) == len(RELEASE_ATTACHMENTS), rows
    for name in RELEASE_ATTACHMENTS:
        digest = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert f"| `{Path(name).name}` | `{digest}` |" in body
    # The step's glob expands distributions in C-locale order, then the fixed attachments follow.
    dists = sorted(name for name in RELEASE_ATTACHMENTS if name.startswith("dist/"))
    fixed = [name for name in RELEASE_ATTACHMENTS if not name.startswith("dist/")]
    attached = "".join(f"{name}\n" for name in [*dists, *fixed])
    output = (tmp_path / "github_output").read_text("utf-8")
    assert output == (
        f"notes_file={tmp_path / 'body.md'}\n"
        f"files<<RELEASE_ATTACHMENTS\n{attached}RELEASE_ATTACHMENTS\n"
    )


def test_release_attaches_exactly_the_files_whose_checksums_it_publishes() -> None:
    """The release step takes its file list from the step that hashes it, and from nowhere else.

    A second hand-written list once attached two security summaries that the body's checksum
    table omitted, so the notes' promise of a SHA-256 for every attachment was false.
    """
    release = _named_workflow("release.yml")["jobs"]["release"]
    notes_index, _ = _step(release, "Release notes for this tag")
    create_index, create = _step(release, "Create GitHub Release")
    assert notes_index < create_index
    assert create["with"]["files"] == "${{ steps.changelog.outputs.files }}"


@pytest.mark.parametrize("name", RELEASE_ATTACHMENTS)
def test_release_body_refuses_a_missing_or_empty_attachment(tmp_path: Path, name: str) -> None:
    """Each attachment must exist with content before the release can be created.

    Parameters
    ----------
    tmp_path : Path
        Working directory for the step runs.
    name : str
        The attachment that is withheld or left empty.
    """
    notes = "# Notes\n"
    hollow = _run_release_notes_step(tmp_path / "hollow", notes, empty=name)
    assert hollow.returncode != 0 and f"missing release attachment: {name}" in hollow.stderr
    if not name.startswith("dist/"):
        # A fixed attachment is named explicitly; a missing one must stop the step. Package
        # distributions come from a glob, whose empty case is the test below.
        absent = _run_release_notes_step(tmp_path / "absent", notes, omit=(name,))
        assert absent.returncode != 0 and f"missing release attachment: {name}" in absent.stderr


def test_release_body_refuses_a_release_without_package_distributions(tmp_path: Path) -> None:
    """No built distribution stops the step instead of releasing only the SBOMs.

    Parameters
    ----------
    tmp_path : Path
        Working directory for the step run.
    """
    dists = tuple(name for name in RELEASE_ATTACHMENTS if name.startswith("dist/"))
    done = _run_release_notes_step(tmp_path, "# Notes\n", omit=dists)
    assert done.returncode != 0 and "no package distributions to attach" in done.stderr


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
    assert sbom_position > _step(job, "Accept the installed Studio the wheel provides")[0]
    assert 'cyclonedx-py environment "$RUNNER_TEMP/installed/bin/python"' in sbom["run"]
    assert "syft scan file:Cargo.lock -o cyclonedx-json=sbom-engine-rust.json" in sbom["run"]
    published = [
        s for s in job["steps"] if "softprops/action-gh-release@" in str(s.get("uses", ""))
    ]
    # The release takes its files from the attachment list of the release-notes step.
    assert published[0]["with"]["files"] == "${{ steps.changelog.outputs.files }}"
    attachments = _step(job, "Release notes for this tag")[1]["run"]
    assert re.search(r"^\s+sbom\.json$", attachments, re.MULTILINE)
    assert re.search(r"^\s+sbom-engine-rust\.json$", attachments, re.MULTILINE)


def test_unpublished_dispatch_runs_the_full_crate_verify_build() -> None:
    """``publish=false`` still proves the crate, while the upload job stays skipped."""
    jobs = _workflow()["jobs"]
    dry_run = jobs["dry-run-crate"]
    assert dry_run["if"] == (
        "github.event_name == 'workflow_dispatch' && github.event.inputs.publish != 'true'"
    )
    commands = _run_text(dry_run)
    assert "cargo publish --manifest-path engine/Cargo.toml --dry-run --locked" in commands
    assert "--no-verify" not in commands
    assert "--dry-run" not in _run_text(jobs["publish-crate"])


def test_container_image_is_scanned_before_any_push() -> None:
    """No image tag reaches GHCR before the CRITICAL/HIGH gate passed, and only the scanned image does.

    The one image build loads into the local daemon. The push step rebuilds nothing: it checks
    that every tag names the scanned image, and only then pushes that image.
    """
    job = _named_workflow("docker.yml")["jobs"]["build-push"]
    build_at, build = _step(job, "Build the image locally")
    scan_at, scan = _step(job, "Trivy fail on CRITICAL/HIGH")
    push_at, push = _step(job, "Push the scanned image")
    assert build_at < scan_at < push_at
    assert build["with"]["load"] is True and build["with"]["push"] is False
    assert "--exit-code 1" in scan["run"]
    assert push["env"]["IMAGE_REF"] == scan["env"]["IMAGE_REF"]
    builds = [
        s
        for s in job["steps"]
        if isinstance(s, dict) and str(s.get("uses", "")).startswith("docker/build-push-action@")
    ]
    assert builds == [build]
    assert not any(
        isinstance(s, dict) and s.get("with", {}).get("push") is True for s in job["steps"]
    )
    commands = push["run"]
    same_image = commands.index(
        'test "$(docker image inspect --format \'{{.Id}}\' "${tag}")" = "${scanned}"'
    )
    # Every tag is checked before the first push, so a mismatch publishes nothing.
    assert (
        commands.index('scanned="$(docker image inspect')
        < same_image
        < commands.index("done", same_image)
        < commands.index('docker push "${tag}"')
    )
    assert _step(job, "Extract metadata")[1]["with"]["flavor"].split() == ["latest=auto"]


_ATTEST = "actions/attest-build-provenance@4d101475d8b20a2381f78447822ac1eab6504dd8"


def test_release_files_and_image_get_signed_build_provenance() -> None:
    """Every Release file and the pushed image carry a Sigstore-signed build provenance.

    The release job attests exactly the files it attaches, under the same condition and
    before the Release is created; the Docker job attests the digest the registry reported
    for the pushed image and stores the attestation in the registry.
    """
    release = _named_workflow("release.yml")["jobs"]["release"]
    assert release["permissions"] == {
        "contents": "write",
        "id-token": "write",
        "attestations": "write",
    }
    notes_at, _ = _step(release, "Release notes for this tag")
    attest_at, attest = _step(release, "Attest the release attachments")
    create_at, create = _step(release, "Create GitHub Release")
    assert notes_at < attest_at < create_at
    assert attest["uses"] == _ATTEST
    assert attest["if"] == create["if"]
    assert attest["with"] == {"subject-path": create["with"]["files"]}

    docker = _named_workflow("docker.yml")["jobs"]["build-push"]
    assert docker["permissions"]["id-token"] == "write"
    assert docker["permissions"]["attestations"] == "write"
    push_at, _ = _step(docker, "Push the scanned image")
    wait_at, _ = _step(docker, "Wait for GHCR image availability")
    digest_at, digest = _step(docker, "Resolve the pushed image digest")
    image_at, image = _step(docker, "Attest the pushed image")
    assert push_at < wait_at < digest_at < image_at
    assert '[[ "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]]' in digest["run"]
    assert image["uses"] == _ATTEST
    assert image["with"] == {
        "subject-name": "${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}",
        "subject-digest": "${{ steps.digest.outputs.digest }}",
        "push-to-registry": True,
    }

    security = (_repo_root() / "SECURITY.md").read_text("utf-8")
    assert "gh attestation verify oci://ghcr.io/anulum/sc-neurocore:<version>" in security
    assert "gh attestation verify sc_neurocore-<version>.tar.gz" in security


def test_release_fuzz_subset_can_import_the_package() -> None:
    """The release sweep's fuzz subset imports the package, so the runtime lock is installed with it."""
    release = _named_workflow("release.yml")["jobs"]["release"]
    tools = _step(release, "Install release security sweep tools")[1]["run"]
    assert "-r requirements/fuzz.txt -r requirements/runtime.txt" in tools


def test_container_build_context_holds_every_wheel_build_input() -> None:
    """Docker copies each source the wheel build hook packages, and does not ignore it.

    ``src/build_support.py`` refuses to build a wheel without the model reference pages and the
    conversion-benchmark sources; a context that drops them fails the image build.
    """
    root = _repo_root()
    hook = (root / "src" / "build_support.py").read_text("utf-8")
    benchmarks = re.findall(r'"((?:bench_|_)ann_to_snn_replay[a-z_]*\.py)"', hook)
    assert sorted(benchmarks) == [
        "_ann_to_snn_replay_measurement.py",
        "_ann_to_snn_replay_profiles.py",
        "bench_ann_to_snn_replay.py",
    ]
    assert '"docs" / "api" / "models"' in hook
    dockerfile = (root / "deploy" / "Dockerfile").read_text("utf-8")
    build_at = dockerfile.index("RUN python -m build --wheel")
    assert dockerfile.index("COPY docs/api/models/ ./docs/api/models/") < build_at
    copy_line = next(line for line in dockerfile.splitlines() if "./benchmarks/" in line)
    assert dockerfile.index(copy_line) < build_at
    ignore = [line.strip() for line in (root / ".dockerignore").read_text("utf-8").splitlines()]
    for kept in ["docs/api/models/", "docs/api/models/*.md"] + [
        f"benchmarks/{b}" for b in benchmarks
    ]:
        assert kept in copy_line or kept.startswith("docs/")
        assert ignore.index(f"!{kept}") > max(
            ignore.index("docs/"), ignore.index("*.md"), ignore.index("benchmarks/")
        )
    for name in benchmarks:
        assert (root / "benchmarks" / name).is_file()
    assert list((root / "docs" / "api" / "models").glob("*.md"))


def test_release_path_builds_resolve_only_the_committed_locks() -> None:
    """Every engine build and Rust tool install on a release path refuses to re-resolve its lock.

    Without ``--locked`` maturin and ``cargo install`` silently pick newer dependency versions than
    the committed lock, so a published wheel or a security tool could differ from what CI tested.
    """
    root = _repo_root()
    sources = [
        root / ".github" / "workflows" / name
        for name in ("publish.yml", "release.yml", "fuzz.yml", "security-scanners.yml")
    ] + [root / "deploy" / "Dockerfile"]
    commands = [
        (source.name, line.strip())
        for source in sources
        for line in source.read_text("utf-8").splitlines()
        if re.search(r"\bmaturin build\b|\bcargo install\b", line)
        and not line.lstrip().startswith("#")
    ]
    assert len(commands) >= 7, commands
    assert [command for command in commands if "--locked" not in command[1]] == []


def test_wheel_workflows_carry_no_step_for_the_untracked_windows_extension() -> None:
    """The Windows extension left the tree in February 2026, so no workflow deletes it any more.

    A fresh checkout cannot contain the file, and no earlier step of these jobs writes it; a step
    that removes it only suggests a tracked binary that the repository no longer has.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "--", "bridge/sc_neurocore_engine/*.pyd"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert tracked == ""
    for name in ("publish.yml", "v3-wheels.yml", "v3-engine.yml"):
        text = (_repo_root() / ".github" / "workflows" / name).read_text("utf-8")
        assert "cp312-win_amd64.pyd" not in text, name
