# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Built distribution model resource contracts

"""Verify actual wheels preserve canonical model descriptors and reference pages."""

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import zipfile

import pytest

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


@pytest.mark.parametrize("through_sdist", [False, True], ids=["wheel", "sdist-wheel"])
def test_wheel_contains_exact_model_resources(
    distribution_source: Path, tmp_path: Path, through_sdist: bool
) -> None:
    """Both supported build paths retain each descriptor and canonical page byte-for-byte."""
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
    with zipfile.ZipFile(wheel_path) as wheel:
        for directory, pattern, destination in (
            ("src/sc_neurocore/neurons/model_descriptors", "*.toml", "neurons/model_descriptors"),
            ("docs/api/models", "*.md", "studio/model_docs"),
        ):
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
    """Replay four model families from the wheel with editable import hooks disabled.

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
    probe = """
import json, sys
from pathlib import Path
installed = Path(sys.argv[1])
sys.path[:0] = [str(installed), sys.argv[2]]
import sc_neurocore
assert Path(sc_neurocore.__file__).is_relative_to(installed)
from sc_neurocore.studio.model_catalogue import model_documentation
from sc_neurocore.neurons.model_catalogue import load_descriptor
from sc_neurocore.studio.replay_pack import replay_pack
packs = json.load(sys.stdin)
for pack in packs:
    name = pack["request"]["name"]
    assert load_descriptor(name) is not None, name
    assert model_documentation(name) is not None, name
    result = replay_pack(pack)
    assert result["verdict"] == "match", result["differences"]
    assert result["worst_state_deviation"] == 0.0
    assert result["runtime_differences"] == []
    pack["expectation"]["spike_count"] += 1
    assert replay_pack(pack)["verdict"] == "mismatch"
print("four model families replay exactly; tampering rejected")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe, str(installed), sysconfig.get_path("purelib")],
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
