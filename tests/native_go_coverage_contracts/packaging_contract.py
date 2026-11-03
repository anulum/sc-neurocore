# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual ordinary distribution diagnostic-isolation contract

"""Build real source and wheel archives without changing the frozen checkout."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import zipfile

from tools.go_mat_coverage_build import NativeGo, digest, save

_DIAGNOSTICS = (
    "src/sc_neurocore/accel/go/mat/coverage_snapshot.go",
    "src/sc_neurocore/accel/go/sc_resetting_mat/coverage_snapshot.go",
    "src/sc_neurocore/accel/go/internal/coverageprofile/snapshot.go",
    "src/sc_neurocore/accel/go/services/mat_simulation_coverage_test.go",
)
_BACKEND = (
    "import os,sys;os.chdir(sys.argv[1]);sys.path.insert(0,sys.argv[1]);"
    "from setuptools import build_meta;"
    "print(getattr(build_meta,sys.argv[2])(sys.argv[3]))"
)


def test_actual_sdist_and_normal_wheel_keep_native_diagnostics_isolated(
    native_builder: NativeGo,
    native_job: Path,
) -> None:
    """Normal setuptools archives cannot silently enroll diagnostic-only native source."""
    repository = native_builder.repository
    original = {name: digest(repository / name) for name in _DIAGNOSTICS}
    view = native_builder.job / "packaging-source"

    def excluded(directory: str, names: list[str]) -> set[str]:
        """Omit build/cache directories while retaining all native source and Studio resources."""
        ignored = {
            name
            for name in names
            if name
            in {
                ".git",
                ".venv",
                ".pixi",
                "__pycache__",
                ".ruff_cache",
                ".mypy_cache",
                ".pytest_cache",
                "node_modules",
                "target",
            }
            or name.endswith(".egg-info")
        }
        if Path(directory) == repository:
            ignored.update(set(names) & {"build", "dist"})
        return ignored

    shutil.copytree(repository, view, ignore=excluded)
    assert all(digest(view / name) == expected for name, expected in original.items())
    outputs = native_builder.job / "artifacts/distributions"
    outputs.mkdir()
    native_builder.require(
        "actual-sdist",
        [sys.executable, "-I", "-B", "-c", _BACKEND, str(view), "build_sdist", str(outputs)],
        view,
        outputs=(outputs,),
    )
    sdists = list(outputs.glob("*.tar.gz"))
    assert len(sdists) == 1
    unpacked = native_builder.job / "unpacked-sdist"
    unpacked.mkdir()
    included: dict[str, bool] = {}
    with tarfile.open(sdists[0]) as archive:
        members = archive.getmembers()
        top = {PurePosixPath(member.name).parts[0] for member in members}
        assert len(top) == 1
        prefix = next(iter(top)) + "/"
        names = {member.name for member in members}
        for name in _DIAGNOSTICS:
            included[name] = prefix + name in names
            if included[name]:
                stream = archive.extractfile(prefix + name)
                assert stream is not None
                data = stream.read()
                assert data == (repository / name).read_bytes()
                assert b"//go:build coverage\n" in data
        for member in members:
            relative = PurePosixPath(member.name)
            assert not relative.is_absolute() and ".." not in relative.parts
            destination = unpacked.joinpath(*relative.parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                assert member.isfile(), "Source archive contains an unqualified special entry"
                destination.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                assert stream is not None
                with destination.open("xb") as output:
                    shutil.copyfileobj(stream, output)
    source = unpacked / prefix.rstrip("/")
    native_builder.require(
        "normal-wheel-from-sdist",
        [sys.executable, "-I", "-B", "-c", _BACKEND, str(source), "build_wheel", str(outputs)],
        source,
        outputs=(outputs,),
    )
    wheels = list(outputs.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        wheel_names = wheel.namelist()
        assert not any(
            member.endswith(name.removeprefix("src/"))
            for member in wheel_names
            for name in _DIAGNOSTICS
        )
    assert original == {name: digest(repository / name) for name in _DIAGNOSTICS}
    before = json.loads((native_job / "source-before.json").read_text())
    assert all(before[str(repository / name)] == value for name, value in original.items())
    save(
        native_job / "artifacts/ordinary-distribution-manifests.json",
        {
            "sdist": {
                "path": str(sdists[0]),
                "sha256": digest(sdists[0]),
                "members": sorted(names),
                "diagnostic_sources_present": included,
            },
            "wheel": {"path": str(wheels[0]), "sha256": digest(wheels[0]), "members": wheel_names},
            "all_diagnostic_source_inputs_unchanged": True,
            "native_source_removed_from_build_view": False,
            "build_cache_directories_excluded": True,
            "whole_distribution_runtime_qualification": False,
        },
    )
