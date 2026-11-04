# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Successful synthesis cache and retained evidence contracts

"""Exercise real mapping reuse, preserved failures and stale evidence refusal."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from filelock import FileLock, Timeout

from tools.function_synthesis_cache import (
    CachedFunctionSynthesis,
    FunctionSynthesisAttemptError,
    FunctionSynthesisCache,
)
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_toolchain import SynthesisToolchain

CaseFixture = tuple[FunctionSynthesisCache, CachedFunctionSynthesis, SynthesisToolchain]


@pytest.fixture(scope="module")
def cache_case(tmp_path_factory: pytest.TempPathFactory) -> CaseFixture:
    """Run one actual zero-cell mapping and retain its complete successful proof."""
    directory = tmp_path_factory.mktemp("real-function-cache")
    tools = SynthesisToolchain.inspect(directory / "toolchain")
    cache = FunctionSynthesisCache(directory / "cache", tools)
    result = cache.measure(SynthesisCase("floor", 8, 0))
    assert not result.reused
    assert result.measurement.resources.total_cells == 0
    return cache, result, tools


def _copy_entry(cache_case: CaseFixture, destination: Path) -> FunctionSynthesisCache:
    _, result, tools = cache_case
    entry = destination / "successful" / result.identity
    shutil.copytree(result.directory, entry)
    return FunctionSynthesisCache(destination, tools)


def test_real_success_is_reused_without_rewriting_native_receipts(cache_case: CaseFixture) -> None:
    """Reuse the source-bound native result and preserve every raw artifact's bytes and time."""
    cache, original, _ = cache_case
    before = {
        path.relative_to(original.directory).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in original.directory.rglob("*")
        if path.is_file()
    }
    requests = sorted(path.name for path in (cache.directory / "attempts").iterdir())
    again = cache.measure(SynthesisCase("floor", 8, 0))
    assert again.reused and again.measurement == original.measurement
    assert again.directory == original.directory and again.identity == original.identity
    assert requests == sorted(path.name for path in (cache.directory / "attempts").iterdir())
    assert again.recorded_at == original.recorded_at
    after = {
        path.relative_to(original.directory).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in original.directory.rglob("*")
        if path.is_file()
    }
    assert before == after
    binding = json.loads((again.directory / "binding.json").read_text())
    assert binding["case"] == "v2/floor/Q8.0"
    assert binding["toolchain"] == cache.tools.manifest()
    assert "src/sc_neurocore/compiler/__init__.py" not in binding["source_sha256"]
    assert "tools/function_synthesis_cache.py" in binding["source_sha256"]
    assert "tools/function_synthesis_checkpoints.py" in binding["source_sha256"]
    assert "tools/function_synthesis_bdd.py" in binding["source_sha256"]
    provenance = cache.source_bindings
    assert provenance == binding["source_sha256"]
    provenance.clear()
    assert cache.source_bindings == binding["source_sha256"]


@pytest.mark.parametrize(
    "artifact",
    [
        "unit.v",
        "flow.ys",
        "mapped.json",
        "mapped.v",
        "stat.json",
        "reference.aig",
        "mapped.aig",
        "yosys.log",
        "yosys.command.json",
        "reference-io.log",
        "reference-io.command.json",
        "mapped-io.log",
        "mapped-io.command.json",
        "equivalence.log",
        "equivalence.command.json",
        "binding.json",
        "measurement.json",
        "execution.txt",
    ],
)
def test_changed_successful_artifact_is_preserved_and_refused(
    cache_case: CaseFixture, tmp_path: Path, artifact: str
) -> None:
    """Reject every changed input/output/command/proof binding without silently rebuilding."""
    cache = _copy_entry(cache_case, tmp_path / "cache")
    entry = cache.directory / "successful" / cache_case[1].identity
    changed = entry / artifact
    if artifact not in {"binding.json", "measurement.json", "execution.txt"}:
        changed = entry / ".retained-artifacts" / (artifact + ".gz")
    with changed.open("ab") as stream:
        stream.write(b"\n")
    corrupted = changed.read_bytes()
    with pytest.raises(ValueError):
        cache.measure(SynthesisCase("floor", 8, 0))
    assert changed.read_bytes() == corrupted
    assert not list((cache.directory / "attempts").iterdir())


@pytest.mark.parametrize(
    "name", ["binding.json", "measurement.json", "execution.txt", "mapped.v", "entry"]
)
def test_linked_successful_evidence_is_refused(
    cache_case: CaseFixture, tmp_path: Path, name: str
) -> None:
    """Refuse linked records even when their target bytes are a real successful mapping."""
    cache = _copy_entry(cache_case, tmp_path / "cache")
    entry = cache.directory / "successful" / cache_case[1].identity
    path = entry if name == "entry" else entry / name
    if name == "mapped.v":
        path = entry / ".retained-artifacts/mapped.v.gz"
    target = tmp_path / "retained"
    path.rename(target)
    path.symlink_to(target, target_is_directory=name == "entry")
    with pytest.raises(ValueError, match="symbolic links"):
        cache.measure(SynthesisCase("floor", 8, 0))
    assert path.is_symlink() and target.exists()


@pytest.mark.parametrize("budget", ["synthesis_timeout", "proof_timeout", "lock_timeout"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_deadline_never_admits_even_a_cached_result(
    cache_case: CaseFixture, budget: str, value: float
) -> None:
    """Reject all nonpositive, nonfinite and Boolean lease/native deadlines before reuse."""
    options = {budget: value}
    with pytest.raises(ValueError, match="finite and positive"):
        cache_case[0].measure(SynthesisCase("floor", 8, 0), **options)


def test_unsupported_legacy_case_creates_no_success_or_attempt(
    cache_case: CaseFixture, tmp_path: Path
) -> None:
    """Preserve unsupported as unavailable instead of a cached zero-cost result."""
    root = tmp_path / "unsupported"
    cache = FunctionSynthesisCache(root, cache_case[2])
    with pytest.raises(ValueError, match="Unsupported functions"):
        cache.measure(SynthesisCase("tan", 16, 8, "lut-v1"))
    assert not root.exists()


def test_real_deadline_retains_attempt_without_success(
    cache_case: CaseFixture, tmp_path: Path
) -> None:
    """Retain actual killed-Yosys receipts while excluding the attempt from successful reuse."""
    cache = FunctionSynthesisCache(tmp_path / "cache", cache_case[2])
    with pytest.raises(FunctionSynthesisAttemptError) as caught:
        cache.measure(SynthesisCase("floor", 8, 0), synthesis_timeout=1e-9)
    assert caught.value.timed_out
    assert not list((cache.directory / "successful").iterdir())
    attempts = list((cache.directory / "attempts").iterdir())
    assert len(attempts) == 1
    assert caught.value.directory == attempts[0]
    assert json.loads((attempts[0] / "request.json").read_text())["case"] == "v2/floor/Q8.0"
    assert (attempts[0] / "measurement/unit.v").is_file()
    assert "timeout: sending signal KILL" in (attempts[0] / "measurement/yosys.log").read_text()


def test_real_same_key_writer_lease_has_a_finite_deadline(cache_case: CaseFixture) -> None:
    """Respect an actual external writer lease instead of reading around its transaction."""
    cache, result, _ = cache_case
    with (
        FileLock(str(cache.directory / "locks" / f"{result.identity}.lock")),
        pytest.raises(Timeout),
    ):
        cache.measure(SynthesisCase("floor", 8, 0), lock_timeout=0.001)
    assert cache.measure(SynthesisCase("floor", 8, 0)).reused


def test_real_formats_and_lowerings_have_distinct_successes(
    cache_case: CaseFixture, tmp_path: Path
) -> None:
    """Keep actual sixteen-bit legacy/new measurements separate from an eight-bit entry."""
    cache = _copy_entry(cache_case, tmp_path / "cache")
    results = [
        cache.measure(SynthesisCase("floor", 16, 8, lowering)) for lowering in ("v2", "lut-v1")
    ]
    assert all(not result.reused for result in results)
    assert len({cache_case[1].identity, *(result.identity for result in results)}) == 3
    assert all(result.measurement.case.input_bits == 16 for result in results)
    assert all(
        result.measurement.to_dict()["proof"]
        == {
            "status": "equivalent",
            "method": "complete-aiger",
            "input_bits": 16,
            "output_bits": 16,
            "latches": 0,
        }
        for result in results
    )


def test_mixed_source_checkout_is_refused_by_a_real_import(tmp_path: Path) -> None:
    """Reject a copied tooling module combined with the original compiler package."""
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "tools").mkdir()
    shutil.copyfile(
        root / "tools/function_synthesis_cache.py", tmp_path / "tools/function_synthesis_cache.py"
    )
    script = tmp_path / "mixed-checkout.py"
    script.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path[:0] = [sys.argv[1], sys.argv[2], sys.argv[3]]\n"
        "from tools.function_synthesis_cache import FunctionSynthesisCache\n"
        "from tools.function_synthesis_toolchain import SynthesisToolchain\n"
        "tools = SynthesisToolchain.inspect(Path(sys.argv[1]) / 'inspection')\n"
        "try:\n"
        "    FunctionSynthesisCache(Path(sys.argv[1]) / 'cache', tools)\n"
        "except ValueError as error:\n"
        "    assert str(error) == 'Compiler and synthesis tools must use the same source checkout'\n"
        "else:\n"
        "    raise AssertionError('Mixed checkout was admitted')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path), str(root / "src"), str(root)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "cache").exists()


def test_real_source_change_during_mapping_cannot_publish_success(tmp_path: Path) -> None:
    """Reject a real source-copy edit after native mapping begins and retain its proof."""
    root = Path(__file__).resolve().parents[2]
    checkout = tmp_path / "source-copy"
    sources = {
        path.relative_to(root).as_posix()
        for path in (root / "src/sc_neurocore").rglob("*.py")
        if path.is_file()
    }
    sources.update(
        path.relative_to(root).as_posix()
        for path in (root / "tools").glob("function_synthesis*.py")
    )
    for name in sorted(sources):
        destination = checkout / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
    script = tmp_path / "source-change.py"
    script.write_text(
        """from pathlib import Path
import sys
import threading
import time

root = Path(sys.argv[1])
sys.path[:0] = [str(root / "src"), str(root)]
from tools.function_synthesis_cache import FunctionSynthesisCache
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_toolchain import SynthesisToolchain

tools = SynthesisToolchain.inspect(root / "inspection")
cache = FunctionSynthesisCache(root / "cache", tools)
changed = threading.Event()

def change_source() -> None:
    \"\"\"Change the actual copied compiler source only after a native attempt starts.\"\"\"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if list((root / "cache/attempts").glob("*/measurement/yosys.command.json")):
            source = root / "src/sc_neurocore/compiler/function_unit_hardware.py"
            with source.open("a") as stream:
                stream.write("\\n# Actual concurrent source change during native mapping.\\n")
            changed.set()
            return
        time.sleep(0.002)

worker = threading.Thread(target=change_source, daemon=True)
worker.start()
try:
    cache.measure(SynthesisCase("floor", 8, 0))
except ValueError as error:
    assert str(error) == "Source checkout changed after cache construction"
else:
    raise AssertionError("A source-changing measurement was cached")
worker.join(timeout=2)
assert changed.is_set()
assert not list((root / "cache/successful").iterdir())
attempts = list((root / "cache/attempts").iterdir())
assert len(attempts) == 1
assert "Networks are equivalent" in (attempts[0] / "measurement/equivalence.log").read_text()
print("Real source change refused after actual successful native proof")
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script), str(checkout)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "source change refused after actual successful native proof" in result.stdout
    changed_source = Path("src/sc_neurocore/compiler/function_unit_hardware.py")
    assert ast.dump(ast.parse((checkout / changed_source).read_text())) == ast.dump(
        ast.parse((root / changed_source).read_text())
    )


def test_lookup_never_launches_a_missing_mapping(cache_case: CaseFixture, tmp_path: Path) -> None:
    """Distinguish an absent success without creating a request, lock or native attempt."""
    root = tmp_path / "missing"
    cache = FunctionSynthesisCache(root, cache_case[2])
    assert cache.get(SynthesisCase("floor", 8, 0)) is None
    assert not root.exists()


def test_lookup_preserves_the_original_measurement_time(cache_case: CaseFixture) -> None:
    """Read a real successful entry with unchanged native counts, binding and UTC time."""
    cache, original, _ = cache_case
    result = cache.get(SynthesisCase("floor", 8, 0))
    assert result is not None
    assert result.reused and result.measurement == original.measurement
    assert result.recorded_at == original.recorded_at
    assert (result.directory / "execution.txt").read_text() == result.recorded_at.isoformat() + "\n"


def test_dangling_success_link_is_preserved_and_refused(
    cache_case: CaseFixture, tmp_path: Path
) -> None:
    """Refuse both lookup and measurement around an existing dangling successful entry."""
    cache = FunctionSynthesisCache(tmp_path / "cache", cache_case[2])
    successful = cache.directory / "successful"
    successful.mkdir(parents=True)
    entry = successful / cache_case[1].identity
    entry.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        cache.get(SynthesisCase("floor", 8, 0))
    with pytest.raises(ValueError, match="symbolic links"):
        cache.measure(SynthesisCase("floor", 8, 0))
    assert entry.is_symlink()


@pytest.mark.parametrize(
    "timestamp",
    [
        "not-a-date\n",
        "2026-10-29T20:00:00\n",
        "2026-10-29T20:00:00+01:00\n",
        "2026-10-29T20:00:00+00:00\n",
    ],
)
def test_replaced_execution_time_cannot_restamp_a_success(
    cache_case: CaseFixture, tmp_path: Path, timestamp: str
) -> None:
    """Reject malformed, non-UTC or newly substituted valid-UTC historical timestamps."""
    cache = _copy_entry(cache_case, tmp_path / "cache")
    execution = cache.directory / "successful" / cache_case[1].identity / "execution.txt"
    execution.write_text(timestamp, encoding="utf-8")
    with pytest.raises(ValueError):
        cache.get(SynthesisCase("floor", 8, 0))
    assert execution.read_text() == timestamp
