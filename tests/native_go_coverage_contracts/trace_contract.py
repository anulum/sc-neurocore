# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual explicit Python trace custody contracts

"""Refuse changed own-code tracing configuration without replacing native providers."""

from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path
import sys

import pytest

from tools.go_coverage_trace import (
    config_text,
    expected_sources,
    union_config,
    validate_config,
    write_bootstrap,
    write_config,
)
from tools.go_mat_coverage_build import NativeGo


def test_actual_trace_configuration_preserves_scope_and_original_input(
    native_job: Path,
    native_repository: Path,
    tmp_path: Path,
) -> None:
    """An explicit trace view preserves real source identity and immutable configuration."""
    config = write_config(native_repository, tmp_path / "trace")
    digest = validate_config(config, native_repository)
    metadata = json.loads((config.parent / "source-platform.json").read_text())
    producer = json.loads((native_job / "producer-inputs.json").read_text())
    original = Path(producer["python_coverage_config"])
    assert validate_config(original, native_repository) == producer["python_coverage_sha256"]
    assert metadata["sources"] == expected_sources(native_repository)
    assert metadata["platform"] == sys.platform and len(digest) == 64
    text = config.read_text()
    assert "branch = true" in text and "patch = _exit" in text
    assert "subprocess" not in text and "omit =" not in text and "fail_under = 100" in text
    with pytest.raises(FileExistsError):
        write_config(native_repository, config.parent)
    assert validate_config(config, native_repository) == digest


@pytest.mark.parametrize(
    "fault", ["scope", "platform", "source", "version", "pin", "repository", "name", "alias"]
)
def test_changed_actual_trace_declarations_fail_closed(
    native_repository: Path,
    tmp_path: Path,
    fault: str,
) -> None:
    """Damaging the genuine writer's output cannot alter coverage scope or its source binding."""
    config = write_config(native_repository, tmp_path / "trace")
    metadata_path = config.parent / "source-platform.json"
    metadata = json.loads(metadata_path.read_text())
    if fault == "scope":
        config.write_text(config.read_text().replace("branch = true", "branch = false"))
    elif fault == "platform":
        metadata["platform"] = "invented-platform"
    elif fault == "source":
        metadata["sources"].pop(next(iter(metadata["sources"])))
    elif fault == "version":
        metadata["coverage_version"] = "0.0.0"
    elif fault == "pin":
        metadata["coverage_pin_sha256"] = "0" * 64
    elif fault == "repository":
        metadata["repository"] = str(tmp_path)
    elif fault == "name":
        other = config.with_name("different.coveragerc")
        config.rename(other)
        config = other
    else:
        alias = tmp_path / "go-mat.coveragerc"
        alias.symlink_to(config)
        config = alias
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        validate_config(config, native_repository)


def test_bootstrap_is_allocation_owned_and_collision_preserves_every_byte(
    native_builder: NativeGo,
) -> None:
    """Generated real child entry points stay in the job and are never overwritten."""
    (native_builder.job / "bootstrap").mkdir()
    preload = write_bootstrap(
        native_builder.job, native_builder.repository, native_builder.python_coverage_config
    )
    files = {p.name: p.read_bytes() for p in preload.parent.iterdir()}
    assert set(files) == {"preload.py", "sitecustomize.py"}
    # REUSE-IgnoreStart
    assert all(
        data.startswith(b"# SPDX-License-Identifier: AGPL-3.0-or-later") for data in files.values()
    )
    # REUSE-IgnoreEnd
    for name, body in files.items():
        ast.parse(body, filename=name)
    assert b"coverage.Coverage(config_file=" in files["sitecustomize.py"]
    assert b"ctypes.CDLL" in files["preload.py"]
    with pytest.raises(FileExistsError):
        write_bootstrap(
            native_builder.job, native_builder.repository, native_builder.python_coverage_config
        )
    assert files == {p.name: p.read_bytes() for p in preload.parent.iterdir()}


def test_actual_single_host_counter_cohort_can_only_bind_its_declared_platform(
    native_job: Path,
    native_repository: Path,
    tmp_path: Path,
) -> None:
    """A real host manifest validates its explicit cohort without claiming a second OS run."""
    producer = json.loads((native_job / "producer-inputs.json").read_text())
    manifest = Path(producer["python_coverage_config"]).parent / "source-platform.json"
    required = frozenset({sys.platform})
    output = union_config(native_repository, tmp_path / "single-cohort", (manifest,), required)
    assert "[paths]" in output.read_text() and native_repository.as_posix() in output.read_text()
    with pytest.raises(ValueError, match="platforms differ"):
        union_config(
            native_repository,
            tmp_path / "missing-real-platform",
            (manifest,),
            frozenset({sys.platform, "not-an-observed-platform"}),
        )
    with pytest.raises(ValueError, match="duplicate platform"):
        union_config(native_repository, tmp_path / "duplicate", (manifest, manifest), required)
    with pytest.raises(ValueError, match="platforms differ"):
        union_config(native_repository, tmp_path / "empty", (), required)


@pytest.mark.parametrize("fault", ["source", "version", "pin", "root"])
def test_changed_actual_union_manifest_cannot_relabel_counter_source(
    native_job: Path,
    native_repository: Path,
    tmp_path: Path,
    fault: str,
) -> None:
    """A damaged copy cannot make counters from different inputs part of the required cohort."""
    producer = json.loads((native_job / "producer-inputs.json").read_text())
    source = Path(producer["python_coverage_config"]).parent / "source-platform.json"
    metadata = json.loads(source.read_text())
    if fault == "source":
        metadata["sources"].pop(next(iter(metadata["sources"])))
    elif fault == "version":
        metadata["coverage_version"] = "0.0.0"
    elif fault == "pin":
        metadata["coverage_pin_sha256"] = "0" * 64
    else:
        metadata["repository"] = None
    copied = tmp_path / "changed.json"
    copied.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        union_config(native_repository, tmp_path / "union", (copied,), frozenset({sys.platform}))


@pytest.mark.parametrize("value", ["ambiguous$path", "ambiguous[path", "ambiguous\npath"])
def test_coverage_paths_cannot_be_interpreted_as_patterns_or_environment_substitutions(
    native_repository: Path,
    tmp_path: Path,
    value: str,
) -> None:
    """Logical path-protocol damage refuses instead of widening own source selection."""
    with pytest.raises(ValueError, match="ambiguous"):
        config_text(native_repository, tmp_path / value)
    with pytest.raises(ValueError, match="ambiguous"):
        config_text(native_repository, tmp_path, (value,))


@pytest.mark.parametrize("fault", ["missing-consumer", "missing-tool", "source-alias"])
def test_missing_or_aliased_actual_source_cohort_cannot_be_traced(
    native_repository: Path, tmp_path: Path, fault: str
) -> None:
    """Damage only an owned copy of the actual cohort; no substitute source can qualify it."""
    view = tmp_path / "source-view"
    sources = expected_sources(native_repository)
    for name in sources:
        destination = view / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(native_repository / name, destination)
    name = (
        next(name for name in sources if name.startswith("tools/"))
        if fault == "missing-tool"
        else "tests/native_go_coverage_consumer.py"
    )
    (view / name).unlink()
    if fault == "source-alias":
        (view / name).symlink_to(native_repository / name)
    if fault == "missing-consumer":
        with pytest.raises(FileNotFoundError, match="native_go_coverage_consumer.py"):
            expected_sources(view)
    else:
        with pytest.raises(ValueError, match="complete actual gate Python source cohort"):
            expected_sources(view)


@pytest.mark.parametrize("fault", ["relative", "parent-alias"])
def test_trace_writer_refuses_actual_noncanonical_output_directory(
    native_repository: Path, tmp_path: Path, fault: str
) -> None:
    """A relative output or real filesystem alias refuses before allocating trace data."""
    if fault == "relative":
        directory = Path(tmp_path.name + "-relative-trace")
    else:
        real = tmp_path / "real-parent"
        real.mkdir()
        alias = tmp_path / "alias-parent"
        alias.symlink_to(real, target_is_directory=True)
        directory = alias / "trace"
    with pytest.raises(ValueError, match="absolute trace allocation"):
        write_config(native_repository, directory)
    assert not directory.exists()


@pytest.mark.parametrize("fault", ["missing-pin", "version-mismatch"])
def test_actual_coverage_package_cannot_satisfy_damaged_requirement_metadata(
    native_repository: Path, tmp_path: Path, fault: str
) -> None:
    """Read the real installed version against a corrupted copy of the real locked declaration."""
    view = tmp_path / "source-view"
    for name in [*expected_sources(native_repository), "requirements/ci-dev.txt"]:
        destination = view / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(native_repository / name, destination)
    requirement = view / "requirements/ci-dev.txt"
    text = requirement.read_text()
    expression = r"^coverage==[^\s]+"
    assert re.search(expression, text, re.MULTILINE) is not None
    replacement = "invalid-declaration" if fault == "missing-pin" else "coverage==0.0.0"
    requirement.write_text(re.sub(expression, replacement, text, count=1, flags=re.MULTILINE))
    directory = tmp_path / "refused-trace"
    with pytest.raises(RuntimeError, match="Actual coverage version differs"):
        write_config(view, directory)
    assert not directory.exists()
