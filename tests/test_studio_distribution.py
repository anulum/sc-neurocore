# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Built distribution model resource contracts

"""Verify actual wheels preserve every declared package resource and reference page."""

from dataclasses import asdict
from fnmatch import fnmatchcase
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tomllib
import zipfile

import pytest

from sc_neurocore.neurons.model_identity import catalogue_counts
from sc_neurocore.studio.codegen import generate_experiment_script, generate_oneliner
from sc_neurocore.studio.experiment_spec import resolve_experiment
from sc_neurocore.studio.replay_pack import build_replay_pack


@pytest.fixture(scope="module")
def distribution_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copy build inputs without editable metadata or modifying the working tree."""
    root = Path(__file__).resolve().parents[1]
    source = tmp_path_factory.mktemp("studio-distribution") / "source"
    source.mkdir()
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", "src", "docs/api/models"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    inputs = set(tracked.stdout.rstrip("\0").split("\0"))
    inputs.update(("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE", "src/build_support.py"))
    for name in sorted(inputs):
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination)
    return source


def build_distribution(
    source: Path, command: str, output: Path
) -> subprocess.CompletedProcess[str]:
    """Invoke the configured setuptools backend in a bounded separate process.

    Parameters
    ----------
    source:
        Disposable source tree containing real project build inputs.
    command:
        Backend operation, either build_wheel or build_sdist.
    output:
        Dedicated distribution output directory.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Captured build status and diagnostics for acceptance or refusal checks.
    """
    return subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os, sys
tracer = None
coverage_file = os.environ.get("STUDIO_BUILD_COVERAGE_FILE")
if coverage_file:
    import coverage
    tracer = coverage.Coverage(
        data_file=coverage_file, data_suffix=True, config_file=False,
        include=["*/src/build_support.py"],
    )
    tracer.start()
try:
    from setuptools import build_meta
    getattr(build_meta, sys.argv[1])(sys.argv[2])
finally:
    if tracer is not None:
        tracer.stop()
        tracer.save()
""",
            command,
            str(output),
        ],
        cwd=source,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def declared_package_resources(source: Path) -> tuple[str, ...]:
    """Return the resource patterns the distribution declares for the package.

    Parameters
    ----------
    source:
        Source tree whose ``pyproject.toml`` is the resource contract.

    Returns
    -------
    tuple of str
        The ``sc_neurocore`` package-data patterns, relative to the package.
    """
    project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
    return tuple(project["tool"]["setuptools"]["package-data"]["sc_neurocore"])


def matches_resource(relative: str, pattern: str) -> bool:
    """Match a package-relative path against one declared pattern, segment by segment.

    Parameters
    ----------
    relative:
        Path inside the package, with ``/`` separators.
    pattern:
        Declared package-data pattern; ``*`` never crosses a directory.

    Returns
    -------
    bool
        Whether the path is one the pattern declares.
    """
    parts, pattern_parts = relative.split("/"), pattern.split("/")
    return len(parts) == len(pattern_parts) and all(
        fnmatchcase(part, expected) for part, expected in zip(parts, pattern_parts, strict=True)
    )


@pytest.mark.parametrize("through_sdist", [False, True], ids=["wheel", "sdist-wheel"])
def test_wheel_contains_exact_model_resources(
    distribution_source: Path, tmp_path: Path, through_sdist: bool
) -> None:
    """Both build paths ship every declared resource and canonical page byte-for-byte.

    Every package-data pattern must match at least one tracked source file, and
    the wheel must hold exactly those files under that pattern -- so a resource
    class the catalogue needs cannot silently drop out of the distribution.
    """
    source = distribution_source
    if through_sdist:
        result = build_distribution(source, "build_sdist", tmp_path / "sdist")
        assert result.returncode == 0, result.stdout + result.stderr
        (archive_path,) = (tmp_path / "sdist").glob("*.tar.gz")
        with tarfile.open(archive_path) as archive:
            archive.extractall(tmp_path / "unpacked", filter="data")
        (source,) = (tmp_path / "unpacked").iterdir()
    result = build_distribution(source, "build_wheel", tmp_path / "wheel")
    assert result.returncode == 0, result.stdout + result.stderr
    (wheel_path,) = (tmp_path / "wheel").glob("*.whl")
    package = source / "src" / "sc_neurocore"
    with zipfile.ZipFile(wheel_path) as wheel:
        members = [name for name in wheel.namelist() if name.startswith("sc_neurocore/")]
        for pattern in declared_package_resources(source):
            declared_files = sorted(
                path.relative_to(package).as_posix()
                for path in package.rglob("*")
                if path.is_file()
                and matches_resource(path.relative_to(package).as_posix(), pattern)
            )
            assert declared_files, f"declared resource pattern {pattern!r} matches no source file"
            shipped = sorted(
                name.removeprefix("sc_neurocore/")
                for name in members
                if matches_resource(name.removeprefix("sc_neurocore/"), pattern)
            )
            assert shipped == declared_files, pattern
            for relative in declared_files:
                assert wheel.read(f"sc_neurocore/{relative}") == (package / relative).read_bytes()
        for directory, pattern, destination in (("docs/api/models", "*.md", "studio/model_docs"),):
            originals = sorted((distribution_source / directory).glob(pattern))
            assert originals, directory
            expected = {f"sc_neurocore/{destination}/{path.name}" for path in originals}
            actual = {
                name for name in wheel.namelist() if name.startswith(f"sc_neurocore/{destination}/")
            }
            assert actual == expected
            for path in originals:
                assert wheel.read(f"sc_neurocore/{destination}/{path.name}") == path.read_bytes()
        installed = tmp_path / "installed"
        wheel.extractall(installed)
    verify_installed_replays(installed, tmp_path)


def verify_installed_replays(installed: Path, workspace: Path) -> None:
    """Execute replay packs, scripts and one-liners with editable hooks disabled.

    Compare every scalar/vector sample, spike, drive digest and state boundary
    for nondefault-timestep, nonconstant-drive, rate and seeded stochastic cases.
    The installed catalogue must also report exactly the counts the source tree
    does, so a receipt-bound identity is receipt-bound only because its receipt
    was shipped and resolved inside the installed package. An offered execution
    lane must run a kernel and agree with the Python lane; a lane the wheel
    does not provide must refuse an explicit request. Dependencies reuse the test interpreter's site-packages; this is package
    isolation, not a fresh dependency-resolution or platform receipt.

    Parameters
    ----------
    installed:
        Extracted wheel root, used before dependency search paths.
    workspace:
        Disposable working directory containing no source checkout.
    """
    requests = [
        {"name": "HodgkinHuxleyNeuron", "dt": 0.05, "duration": 5.0, "protocol": "step"},
        {"name": "PoissonNeuron", "params": {"seed": 77, "rate_hz": 150}, "duration": 5.0},
        {"name": "WilsonCowanUnit", "duration": 5.0},
        {"name": "AmariNeuralField", "duration": 5.0},
    ]
    packs = [build_replay_pack(request) for request in requests]
    (workspace / "packs.json").write_text(json.dumps(packs), encoding="utf-8")
    for index, pack in enumerate(packs):
        spec = resolve_experiment(pack["request"])
        for form, generator in (
            ("script", generate_experiment_script),
            ("oneliner", generate_oneliner),
        ):
            (workspace / f"{index}-{form}.py").write_text(
                generator(spec, pack["request"]), encoding="utf-8"
            )
    probe = """
import contextlib, io, json, runpy, sys
from pathlib import Path
installed = Path(sys.argv[1])
sys.path[:0] = [str(installed), sys.argv[2]]
import sc_neurocore
assert Path(sc_neurocore.__file__).is_relative_to(installed)
from sc_neurocore.studio.model_catalogue import model_documentation
from sc_neurocore.neurons.model_catalogue import load_descriptor
from sc_neurocore.studio.replay_pack import replay_pack, replay_expectation
from dataclasses import asdict
from sc_neurocore.neurons.model_identity import catalogue_counts, iter_source_catalogue
from sc_neurocore.neurons.model_receipts import RECEIPT_DIRECTORY
assert RECEIPT_DIRECTORY.is_relative_to(installed)
counts = asdict(catalogue_counts())
assert counts == json.loads(sys.argv[3]), counts
bound = [record for record in iter_source_catalogue() if record.revalidation == "receipt-bound"]
assert len(bound) == counts["receipt_bound_complete"] > 0
from sc_neurocore.runtime_lanes import ACCEL_ROOT, lane_statuses
assert ACCEL_ROOT.is_relative_to(installed)
lanes = {status.lane: status for status in lane_statuses()}
assert lanes["python"].resources_present
# The wheel ships no Julia kernels and no built Go or Mojo libraries, even
# where juliacall itself is importable.
assert not any(lanes[lane].resources_present for lane in ("julia", "go", "mojo")), lanes
# Each lane the installation offers runs; a lane it lacks is refused, never
# silently replaced by Python.
import numpy as np
from sc_neurocore.accel.alpha import simulate_alpha
drive = [0.3] * 40 + [1.4] * 40
reference = simulate_alpha(exc_current=drive, backend="python")
assert int(np.sum(reference["spikes"])) > 0
if lanes["rust"].resources_present:
    native = simulate_alpha(exc_current=drive, backend="rust")
    np.testing.assert_allclose(native["v"], reference["v"], rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(native["spikes"], reference["spikes"])
    refused = ("go", "mojo")
else:
    refused = ("rust", "go", "mojo")
for lane in refused:
    try:
        simulate_alpha(exc_current=drive, backend=lane)
    except RuntimeError as refusal:
        assert "unavailable" in str(refusal)
    else:
        raise AssertionError(f"the {lane} lane ran without its resources")
packs = json.load(sys.stdin)
for index, pack in enumerate(packs):
    name = pack["request"]["name"]
    assert load_descriptor(name) is not None, name
    assert model_documentation(name) is not None, name
    result = replay_pack(pack)
    assert result["verdict"] == "match", result["differences"]
    assert result["worst_state_deviation"] == 0.0
    assert result["runtime_differences"] == []
    for form, result_name in (("script", "result"), ("oneliner", "r")):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            namespace = runpy.run_path(f"{index}-{form}.py", run_name="__main__")
        assert "spikes in" in output.getvalue(), output.getvalue()
        assert replay_expectation(namespace[result_name]) == pack["expectation"], (name, form)
        if form == "script":
            assert namespace["REQUEST"] == pack["request"]
            assert namespace["EXPERIMENT_SHA256"] == pack["experiment_sha256"]
    pack["expectation"]["spike_count"] += 1
    assert replay_pack(pack)["verdict"] == "mismatch"
print("four model families replay exactly; both script forms reproduce full traces; tampering rejected")
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            probe,
            str(installed),
            sysconfig.get_path("purelib"),
            json.dumps(asdict(catalogue_counts())),
        ],
        cwd=workspace,
        input=json.dumps(packs),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "four model families replay exactly" in result.stdout


def test_build_refuses_missing_model_pages(distribution_source: Path, tmp_path: Path) -> None:
    """A source tree without reference pages must fail rather than ship an undocumented Studio."""
    source = tmp_path / "source"
    shutil.copytree(
        distribution_source, source, ignore=shutil.ignore_patterns("build", "*.egg-info")
    )
    pages = source / "docs/api/models"
    pages.rename(source / "withheld-model-pages")
    result = build_distribution(source, "build_wheel", tmp_path / "wheel")
    assert result.returncode != 0
    assert "No model reference pages" in result.stdout + result.stderr


def test_rebuild_replaces_modified_and_removed_pages(
    distribution_source: Path, tmp_path: Path
) -> None:
    """Reused build directories must not preserve deleted pages or older-timestamp content."""
    source = tmp_path / "source"
    shutil.copytree(
        distribution_source, source, ignore=shutil.ignore_patterns("build", "*.egg-info")
    )
    first = build_distribution(source, "build_wheel", tmp_path / "first")
    assert first.returncode == 0, first.stdout + first.stderr
    pages = sorted((source / "docs/api/models").glob("*.md"))
    removed, changed = pages[:2]
    removed.rename(source / removed.name)
    changed.write_text("# Revised reference page\n", encoding="utf-8")
    os.utime(changed, (1, 1))
    second = build_distribution(source, "build_wheel", tmp_path / "second")
    assert second.returncode == 0, second.stdout + second.stderr
    (wheel_path,) = (tmp_path / "second").glob("*.whl")
    with zipfile.ZipFile(wheel_path) as wheel:
        assert f"sc_neurocore/studio/model_docs/{removed.name}" not in wheel.namelist()
        assert wheel.read(f"sc_neurocore/studio/model_docs/{changed.name}") == changed.read_bytes()
