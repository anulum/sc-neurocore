# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real native proof retention and corrupt storage refusal

"""Read complete native proofs after lossless retention and reject corrupted custody."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path

import pytest

from tools.function_synthesis_cache import CachedFunctionSynthesis, FunctionSynthesisCache
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_retention import materialize_artifacts, retain_artifacts
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_yosys import read_function_synthesis_measurement

RetentionCase = tuple[FunctionSynthesisCache, CachedFunctionSynthesis, SynthesisToolchain]


@pytest.fixture(scope="module")
def retention_case(tmp_path_factory: pytest.TempPathFactory) -> RetentionCase:
    """Acquire an actual native mapping and its complete retained equivalence proof."""
    root = tmp_path_factory.mktemp("native-retained-proof")
    tools = SynthesisToolchain.inspect(root / "toolchain")
    cache = FunctionSynthesisCache(root / "cache", tools)
    result = cache.measure(SynthesisCase("floor", 8, 0))
    return cache, result, tools


def _copy_case(retention_case: RetentionCase, root: Path) -> tuple[FunctionSynthesisCache, Path]:
    """Copy a real successful store for one independently damaged public cache lookup."""
    _, result, tools = retention_case
    entry = root / "successful" / result.identity
    shutil.copytree(result.directory, entry)
    return FunctionSynthesisCache(root, tools), entry


def test_complete_native_proof_survives_lossless_retention(retention_case: RetentionCase) -> None:
    """Revalidate every original native proof byte and remove the temporary view on exit."""
    cache, result, tools = retention_case
    stored = {
        path.relative_to(result.directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in result.directory.rglob("*")
        if path.is_file()
    }
    with result.open_artifacts() as view:
        assert view != result.directory and view.is_dir()
        assert view.parent == result.directory.parent
        assert (view / "execution.txt").read_text() == result.recorded_at.isoformat() + "\n"
        assert (
            read_function_synthesis_measurement(SynthesisCase("floor", 8, 0), view, tools)
            == result.measurement
        )
        for name, identity in result.measurement.artifact_sha256:
            assert hashlib.sha256((view / name).read_bytes()).hexdigest() == identity
        manifest = json.loads((result.directory / "retained-artifacts.json").read_text())
        assert sum(row["stored_bytes"] for row in manifest["artifacts"].values()) < sum(
            row["bytes"] for row in manifest["artifacts"].values()
        )
    assert not view.exists()
    assert stored == {
        path.relative_to(result.directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in result.directory.rglob("*")
        if path.is_file()
    }
    again = cache.get(SynthesisCase("floor", 8, 0))
    assert again is not None and again.measurement == result.measurement
    assert again.recorded_at == result.recorded_at


def test_nested_native_dsp_proof_is_admitted_after_restoration(
    retention_case: RetentionCase, tmp_path: Path
) -> None:
    """Revalidate actual nested arithmetic and residual proof artifacts after complete DSP mapping."""
    tools = retention_case[2]
    cache = FunctionSynthesisCache(tmp_path / "cache", tools)
    case = SynthesisCase("sin", 8, 4)
    result = cache.measure(case)
    assert result.measurement.proof_method == "compositional"
    with result.open_artifacts() as view:
        assert (view / "composition-proof/proof-record.json").is_file()
        assert read_function_synthesis_measurement(case, view, tools) == result.measurement
    again = cache.get(case)
    assert again is not None and again.measurement == result.measurement
    assert again.recorded_at == result.recorded_at


@pytest.mark.parametrize(
    "damage",
    ["missing-blob", "extra-raw-file", "bad-blob", "linked-blob", "partial-store", "invalid-json"],
)
def test_corrupt_native_store_never_becomes_a_cache_hit(
    retention_case: RetentionCase, tmp_path: Path, damage: str
) -> None:
    """Reject missing, changed, linked, ambiguous and incomplete real native proof storage."""
    cache, entry = _copy_case(retention_case, tmp_path / "cache")
    blob = entry / ".retained-artifacts/mapped.v.gz"
    manifest = entry / "retained-artifacts.json"
    if damage == "missing-blob":
        blob.unlink()
    elif damage == "extra-raw-file":
        (entry / "mapped.v").write_bytes(b"unqualified replacement")
    elif damage == "bad-blob":
        with blob.open("ab") as stream:
            stream.write(b"changed")
    elif damage == "linked-blob":
        target = tmp_path / "original-blob"
        blob.rename(target)
        blob.symlink_to(target)
    elif damage == "partial-store":
        manifest.unlink()
    else:
        manifest.write_text("{not a manifest")
    with pytest.raises(ValueError):
        cache.get(SynthesisCase("floor", 8, 0))
    assert not (cache.directory / "attempts").exists()


@pytest.mark.parametrize(
    "path",
    [
        "../escape",
        "/escape",
        "./mapped.v",
        "x/../mapped.v",
        "binding.json",
        ".retained-artifacts/x",
    ],
)
def test_storage_manifest_cannot_redirect_original_artifact_paths(
    retention_case: RetentionCase, tmp_path: Path, path: str
) -> None:
    """Refuse traversal, absolute, noncanonical and reserved native restoration destinations."""
    cache, entry = _copy_case(retention_case, tmp_path / "cache")
    manifest = entry / "retained-artifacts.json"
    value = json.loads(manifest.read_text())
    value["artifacts"][path] = value["artifacts"].pop("mapped.v")
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        cache.get(SynthesisCase("floor", 8, 0))
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bytes", True),
        ("bytes", -1),
        ("stored_bytes", 0),
        ("mode", 0o4000),
        ("sha256", "x"),
        ("stored_sha256", 1),
    ],
)
def test_invalid_storage_identities_are_refused_before_restoration(
    retention_case: RetentionCase, tmp_path: Path, field: str, value: object
) -> None:
    """Bound original data and retain exact integer, mode and digest contracts."""
    cache, entry = _copy_case(retention_case, tmp_path / "cache")
    manifest = entry / "retained-artifacts.json"
    document = json.loads(manifest.read_text())
    document["artifacts"]["mapped.v"][field] = value
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        cache.get(SynthesisCase("floor", 8, 0))


@pytest.mark.parametrize(
    "damage", ["truncated", "short-original", "long-original", "forged-native-output"]
)
def test_valid_storage_checksums_cannot_replace_complete_native_proof(
    retention_case: RetentionCase, tmp_path: Path, damage: str
) -> None:
    """Reject corrupt streams, bounded-out decoding and changed native outputs after custody checks."""
    cache, entry = _copy_case(retention_case, tmp_path / "cache")
    manifest = entry / "retained-artifacts.json"
    document = json.loads(manifest.read_text())
    record = document["artifacts"]["mapped.v"]
    blob = entry / ".retained-artifacts/mapped.v.gz"
    if damage == "truncated":
        blob.write_bytes(blob.read_bytes()[:-8])
    elif damage == "short-original":
        record["bytes"] = 1
    elif damage == "long-original":
        record["bytes"] += 1
    else:
        changed = gzip.decompress(blob.read_bytes()) + b"\n"
        blob.write_bytes(gzip.compress(changed, mtime=0))
        record["sha256"] = hashlib.sha256(changed).hexdigest()
        record["bytes"] = len(changed)
    record["stored_bytes"] = blob.stat().st_size
    record["stored_sha256"] = hashlib.sha256(blob.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        cache.get(SynthesisCase("floor", 8, 0))


def test_original_file_modes_and_legacy_native_layout_remain_readable(
    retention_case: RetentionCase, tmp_path: Path
) -> None:
    """Admit a real legacy acquisition and restore its deliberately executable native script exactly."""
    _, result, tools = retention_case
    plain = tmp_path / "plain"
    with result.open_artifacts() as view:
        shutil.copytree(view, plain)
    with materialize_artifacts(plain) as legacy:
        assert legacy == plain
        assert (
            read_function_synthesis_measurement(SynthesisCase("floor", 8, 0), legacy, tools)
            == result.measurement
        )
    (plain / "flow.ys").chmod(0o750)
    original = (plain / "flow.ys").read_bytes()
    retain_artifacts(plain)
    assert not (plain / "flow.ys").exists()
    with materialize_artifacts(plain) as restored:
        assert (restored / "flow.ys").read_bytes() == original
        assert (restored / "flow.ys").stat().st_mode & 0o777 == 0o750
    with pytest.raises(ValueError):
        retain_artifacts(plain)


@pytest.mark.parametrize("damage", ["linked-directory", "linked-file", "pipe"])
def test_native_retention_refuses_nonregular_acquisitions(
    retention_case: RetentionCase, tmp_path: Path, damage: str
) -> None:
    """Keep real native proof bytes from being substituted by links or special files."""
    _, result, _ = retention_case
    plain = tmp_path / "plain"
    with result.open_artifacts() as view:
        shutil.copytree(view, plain)
    if damage == "linked-directory":
        target = tmp_path / "real"
        plain.rename(target)
        plain.symlink_to(target, target_is_directory=True)
    elif damage == "linked-file":
        target = tmp_path / "native-flow"
        (plain / "flow.ys").rename(target)
        (plain / "flow.ys").symlink_to(target)
    else:
        os.mkfifo(plain / "native-pipe")
    with pytest.raises(ValueError):
        retain_artifacts(plain)


@pytest.mark.parametrize("damage", ["manifest-size", "schema", "record", "aggregate-size"])
def test_invalid_native_manifest_is_refused_before_data_expansion(
    retention_case: RetentionCase, tmp_path: Path, damage: str
) -> None:
    """Reject unbounded or malformed retention metadata before restoring native data."""
    cache, entry = _copy_case(retention_case, tmp_path / "cache")
    manifest = entry / "retained-artifacts.json"
    document = json.loads(manifest.read_text())
    if damage == "manifest-size":
        with manifest.open("r+b") as stream:
            stream.truncate(16 * 1024**2 + 1)
    else:
        if damage == "schema":
            document["schema_version"] = "unqualified-format"
        elif damage == "record":
            document["artifacts"]["mapped.v"] = []
        else:
            for record in document["artifacts"].values():
                record["bytes"] = 1024**3
        manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        cache.get(SynthesisCase("floor", 8, 0))


@pytest.mark.parametrize("damage", ["no-native-files", "oversized-native-file"])
def test_native_storage_refuses_empty_or_unbounded_acquisitions(
    retention_case: RetentionCase, tmp_path: Path, damage: str
) -> None:
    """Refuse absent native evidence or actual native bytes above a caller's finite budget."""
    _, result, _ = retention_case
    plain = tmp_path / "plain"
    with result.open_artifacts() as view:
        shutil.copytree(view, plain)
    if damage == "no-native-files":
        for path in plain.rglob("*"):
            if path.is_file() and path.name not in {
                "binding.json",
                "measurement.json",
                "execution.txt",
            }:
                path.unlink()
    native_bytes = sum(
        path.stat().st_size
        for path in plain.rglob("*")
        if path.is_file() and path.name not in {"binding.json", "measurement.json", "execution.txt"}
    )
    byte_budget = native_bytes - 1 if damage == "oversized-native-file" else 1024**3
    before = {
        path.relative_to(plain).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in plain.rglob("*")
        if path.is_file()
    }
    with pytest.raises(ValueError):
        retain_artifacts(plain, byte_budget=byte_budget)
    assert not (plain / ".retained-artifacts").exists()
    assert before == {
        path.relative_to(plain).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in plain.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-11-01T00:00:00\n",
        "2026-11-01T00:00:00+01:00\n",
        "2026-11-01T00:00:00+00:00",
        "2026-11-01T00:00:00+00:00\n",
    ],
)
def test_storage_metadata_cannot_rewrite_native_acquisition_time(
    retention_case: RetentionCase, tmp_path: Path, timestamp: str
) -> None:
    """Keep canonical UTC and the original proof-bound time after coherent storage metadata edits."""
    cache, entry = _copy_case(retention_case, tmp_path / "cache")
    (entry / "execution.txt").write_text(timestamp)
    manifest = entry / "retained-artifacts.json"
    document = json.loads(manifest.read_text())
    document["metadata_sha256"]["execution.txt"] = hashlib.sha256(timestamp.encode()).hexdigest()
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        cache.get(SynthesisCase("floor", 8, 0))
    assert (entry / "execution.txt").read_text() == timestamp


@pytest.mark.parametrize(
    "damage",
    ["metadata-disappear", "native-change", "metadata-change", "view-change", "store-change"],
)
def test_concurrent_filesystem_change_never_admits_native_proof(
    retention_case: RetentionCase, tmp_path: Path, damage: str
) -> None:
    """Plant actual filesystem changes at real file-open events and require public admission to fail."""
    _, result, _ = retention_case
    packing = damage in {"metadata-disappear", "native-change", "metadata-change"}
    directory = tmp_path / "acquisition"
    if packing:
        with result.open_artifacts() as view:
            shutil.copytree(view, directory)
    else:
        shutil.copytree(result.directory, directory)
    armed = True
    restored: Path | None = None

    def change_files(event: str, arguments: tuple[object, ...]) -> None:
        """Modify real test-owned files once, without replacing any production operation."""
        nonlocal armed, restored
        if not armed:
            return
        if event == "tempfile.mkdtemp":
            restored = Path(str(arguments[0]))
        if event != "open":
            return
        path = Path(str(arguments[0]))
        reading = arguments[1] == "r"
        if damage == "metadata-disappear" and path == directory / "binding.json":
            armed = False
            (directory / "execution.txt").unlink()
        elif path == directory / ".retained-artifacts/mapped.v.gz" and reading:
            armed = False
            if damage == "native-change":
                with (directory / "mapped.v").open("ab") as stream:
                    stream.write(b"\n")
            elif damage == "metadata-change":
                with (directory / "execution.txt").open("a") as stream:
                    stream.write("\n")
            elif damage == "view-change":
                assert restored is not None
                with (restored / "execution.txt").open("a") as stream:
                    stream.write("\n")
            else:
                manifest = directory / "retained-artifacts.json"
                document = json.loads(manifest.read_text())
                document["artifacts"]["mapped.v"]["mode"] ^= 0o100
                manifest.write_text(json.dumps(document))

    # CPython requires this opt-in to trace Python audit callbacks.
    change_files.__dict__["__cantrace__"] = True
    sys.addaudithook(change_files)
    try:
        with pytest.raises(ValueError), ExitStack() as stack:
            if packing:
                retain_artifacts(directory)
            else:
                stack.enter_context(materialize_artifacts(directory))
        assert not armed
    finally:
        armed = False


@pytest.mark.parametrize("damage", ["duplicate-key", "NaN", "Infinity", "overflow", "nesting"])
def test_ambiguous_or_unbounded_native_manifest_is_refused_readonly(
    retention_case: RetentionCase, tmp_path: Path, damage: str
) -> None:
    """Reject ambiguous and nonfinite storage metadata through actual successful cache lookup."""
    cache, entry = _copy_case(retention_case, tmp_path / "cache")
    manifest = entry / "retained-artifacts.json"
    original = manifest.read_bytes()
    if damage == "duplicate-key":
        payload = b'{"schema_version":"contradictory",' + original.lstrip()[1:]
    elif damage == "nesting":
        payload = b"[" * 2000 + b"0" + b"]" * 2000
    else:
        token = "1e999" if damage == "overflow" else damage
        payload = b'{"unexpected":' + token.encode() + b"," + original.lstrip()[1:]
    manifest.write_bytes(payload)
    before = manifest.read_bytes(), manifest.stat().st_mtime_ns
    with pytest.raises(ValueError):
        cache.get(SynthesisCase("floor", 8, 0))
    assert (manifest.read_bytes(), manifest.stat().st_mtime_ns) == before
    assert not (cache.directory / "attempts").exists()


@pytest.mark.parametrize("budget", [True, 0, -1, 1024**3 + 1])
def test_invalid_caller_byte_budget_preserves_genuine_native_artifacts(
    retention_case: RetentionCase, tmp_path: Path, budget: int
) -> None:
    """Refuse invalid or enlarged caller limits without altering an actual completed acquisition."""
    _, result, _ = retention_case
    plain = tmp_path / "plain"
    with result.open_artifacts() as view:
        shutil.copytree(view, plain)
    before = {
        p.relative_to(plain).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in plain.rglob("*")
        if p.is_file()
    }
    with pytest.raises(ValueError, match="integer byte budget"):
        retain_artifacts(plain, byte_budget=budget)
    assert before == {
        p.relative_to(plain).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in plain.rglob("*")
        if p.is_file()
    }
    assert not (plain / ".retained-artifacts").exists()


def test_exact_actual_native_byte_budget_retains_the_complete_proof(
    retention_case: RetentionCase, tmp_path: Path
) -> None:
    """Admit the exact original byte boundary and reconstruct every real proof artifact unchanged."""
    _, result, tools = retention_case
    plain = tmp_path / "plain"
    with result.open_artifacts() as view:
        shutil.copytree(view, plain)
    native_bytes = sum(
        p.stat().st_size
        for p in plain.rglob("*")
        if p.is_file() and p.name not in {"binding.json", "measurement.json", "execution.txt"}
    )
    assert 0 < native_bytes < 1024**3
    retain_artifacts(plain, byte_budget=native_bytes)
    with materialize_artifacts(plain) as restored:
        assert (
            read_function_synthesis_measurement(SynthesisCase("floor", 8, 0), restored, tools)
            == result.measurement
        )
