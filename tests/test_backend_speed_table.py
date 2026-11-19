# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the packaged measured backend order generator

"""The packaged order table holds current or reviewed measurements only, and names every record it leaves out."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tools import generate_backend_speed_table as table

REPO_ROOT = Path(__file__).resolve().parents[1]

_CPU = "Fixture CPU 1 @ 1.00GHz"


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _record(
    kernel: str | None, sources: dict[str, bytes] | None, *, timed: bool = True
) -> dict[str, object]:
    backends: dict[str, object] = {
        "rust": {"available": True, "used": True, "median_call_ms": 2.0 if timed else None},
        "go": {"available": True, "used": True, "median_call_ms": 1.0 if timed else None},
        "python": {"available": True, "used": True, "median_call_ms": 9.0 if timed else None},
    }
    payload: dict[str, object] = {"backends": backends, "meta": {"cpu": _CPU}}
    if kernel is not None:
        payload["kernel"] = kernel
    if sources is not None:
        payload["source_hashes"] = {
            path: hashlib.sha256(raw).hexdigest() for path, raw in sources.items()
        }
    return payload


@pytest.fixture
def fixture_tree(tmp_path: Path) -> Path:
    """Build a work tree with one record for every reason, two in a reviewed history."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "benchmarks" / "results").mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.name", "fixture")
    _git(root, "config", "user.email", "fixture@invalid")
    measured = {
        "src/current_kernel.py": b"def run():\n    return 1\n",
        "src/changed_kernel.py": b"def run():\n    return 2\n",
        "src/documented_kernel.py": b'def run():\n    """Old words."""\n    return 3\n',
        "src/twin_kernel.py": b"def run():\n    return 4\n",
    }
    for path, raw in measured.items():
        (root / path).write_bytes(raw)
    records = {
        "bench_current.json": _record(
            "current_k", {"src/current_kernel.py": measured["src/current_kernel.py"]}
        ),
        "bench_outdated.json": _record(
            "outdated_k", {"src/changed_kernel.py": measured["src/changed_kernel.py"]}
        ),
        "bench_reviewed.json": _record(
            "reviewed_k", {"src/documented_kernel.py": measured["src/documented_kernel.py"]}
        ),
        "bench_unbound.json": _record("unbound_k", None),
        "bench_shape.json": _record(
            None, {"src/current_kernel.py": measured["src/current_kernel.py"]}
        ),
        "bench_untimed.json": _record(
            "untimed_k", {"src/current_kernel.py": measured["src/current_kernel.py"]}, timed=False
        ),
        "bench_twin_a.json": _record(
            "twin_k", {"src/twin_kernel.py": measured["src/twin_kernel.py"]}
        ),
        "bench_twin_b.json": _record(
            "twin_k", {"src/twin_kernel.py": measured["src/twin_kernel.py"]}
        ),
    }
    for name, payload in records.items():
        (root / "benchmarks" / "results" / name).write_text(json.dumps(payload), encoding="utf-8")
    (root / "benchmarks" / "results" / "not_a_bench.json").write_text("{}", encoding="utf-8")
    (root / "benchmarks" / "results" / "bench_untracked.json").write_text(
        json.dumps(_record("untracked_k", None)), encoding="utf-8"
    )
    _git(
        root,
        "add",
        "src",
        *(f"benchmarks/results/{name}" for name in records),
        "benchmarks/results/not_a_bench.json",
    )
    _git(root, "commit", "-q", "-m", "measured")
    (root / "src/changed_kernel.py").write_bytes(b"def run():\n    return 20\n")
    documented = b'def run():\n    """New words."""\n    return 3\n'
    (root / "src/documented_kernel.py").write_bytes(documented)
    (root / "benchmarks" / "source_change_reviews.toml").write_text(
        "schema_version = 1\n\n[[review]]\n"
        'record = "bench_reviewed.json"\n'
        'path = "src/documented_kernel.py"\n'
        f'source_sha256 = "{hashlib.sha256(documented).hexdigest()}"\n'
        'reason = "Only a docstring changed."\n',
        encoding="utf-8",
    )
    _git(root, "add", "src", "benchmarks/source_change_reviews.toml")
    _git(root, "commit", "-q", "-m", "changes after the measurement")
    return root


def test_the_committed_table_equals_a_fresh_generation() -> None:
    """The packaged table is regenerated whenever a record or a bound source changes; drift fails here."""
    committed = (REPO_ROOT / table.DEFAULT_OUTPUT).read_text(encoding="utf-8")
    assert committed == table.render(table.build(REPO_ROOT))


def test_the_committed_table_admits_only_current_or_reviewed_records() -> None:
    """No entry rests on an outdated or unbound record, and every record left out has its reason."""
    built = table.build(REPO_ROOT)
    assert all(entry["currency"] in table.ADMITTED for entry in built["entries"])
    reasons = {"shape", "outdated", "unbound", "no_timed_backend", "duplicate"}
    assert all(item["reason"] in reasons for item in built["left_out"])
    named = {entry["record"] for entry in built["entries"]} | {
        item["record"] for item in built["left_out"]
    }
    assert len(named) == len(built["entries"]) + len(built["left_out"])


def test_each_record_of_the_fixture_tree_gets_its_rule(fixture_tree: Path) -> None:
    """Current and reviewed records enter; every other tracked record is listed with its own reason."""
    built = table.build(fixture_tree)
    assert [(entry["record"], entry["currency"]) for entry in built["entries"]] == [
        ("bench_current.json", "current"),
        ("bench_reviewed.json", "reviewed"),
    ]
    current = built["entries"][0]
    assert current["kernel"] == "current_k" and current["cpu_model"] == _CPU
    assert current["isa_level"] == table.UNRECORDED
    assert current["order"] == ["go", "rust", "python"]
    raw = (fixture_tree / "benchmarks/results/bench_current.json").read_bytes()
    assert current["record_sha256"] == hashlib.sha256(raw).hexdigest()
    assert list(current["source_digests"]) == ["src/current_kernel.py"]
    assert built["left_out"] == [
        {"record": "bench_outdated.json", "reason": "outdated"},
        {"record": "bench_shape.json", "reason": "shape"},
        {"record": "bench_twin_a.json", "reason": "duplicate"},
        {"record": "bench_twin_b.json", "reason": "duplicate"},
        {"record": "bench_unbound.json", "reason": "unbound"},
        {"record": "bench_untimed.json", "reason": "no_timed_backend"},
    ]


def test_the_command_writes_checks_and_summarises(
    fixture_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--write`` then ``--check`` passes; an edited table fails ``--check``; ``--summary`` counts by reason."""
    output = fixture_tree / "table.json"
    arguments = ["--root", str(fixture_tree), "--output", str(output)]
    assert table.main([*arguments, "--check"]) == 1
    assert table.main([*arguments, "--write", "--summary"]) == 0
    assert (
        "entries 2; left out 6 (duplicate 2, no_timed_backend 1, outdated 1, shape 1, unbound 1)"
        in (capsys.readouterr().out)
    )
    assert table.main([*arguments, "--check"]) == 0
    output.write_text(
        output.read_text(encoding="utf-8").replace("current_k", "edited_k"), encoding="utf-8"
    )
    assert table.main([*arguments, "--check"]) == 1


def test_a_root_outside_a_work_tree_is_refused(tmp_path: Path) -> None:
    """Only tracked records are read; a plain directory gives exit code 2."""
    assert table.main(["--root", str(tmp_path), "--check"]) == 2
    with pytest.raises(ValueError, match="not the top of a Git work tree"):
        table.build(tmp_path)


def test_speed_order_keeps_only_lanes_that_ran_fastest_first() -> None:
    """Unavailable, unused or untimed lanes order nothing; ties keep the record's order."""
    backends = {
        "rust": {"available": True, "used": True, "median_call_ms": 1.5},
        "go": {"available": True, "used": True, "median_call_ms": 0.5},
        "mojo": {"available": False, "used": False, "median_call_ms": 0.1},
        "julia": {"available": True, "used": True, "median_call_ms": None},
        "boolean": {"available": True, "used": True, "median_call_ms": True},
        "twin": {"available": True, "used": True, "median_call_ms": 1.5},
        "python": {"available": True, "used": True, "median_call_ms": 9.0},
        "stray": "not-a-dict",
    }
    assert table.speed_order(backends) == ["go", "rust", "twin", "python"]


def test_isa_level_comes_from_recorded_flags_only() -> None:
    """A record that states its host's flags gets their level; any other record is unrecorded."""
    baseline = "cmov cx8 fpu fxsr mmx syscall sse sse2"
    v2 = "cx16 lahf_lm popcnt pni sse4_1 sse4_2 ssse3"
    v3 = "avx avx2 bmi1 bmi2 f16c fma abm movbe"
    flags = f"{baseline} {v2} {v3}".split()
    assert table.isa_level({"cpu": _CPU, "cpu_flags": flags}) == "x86-64-v3"
    assert table.isa_level({"cpu": _CPU}) == table.UNRECORDED
    assert table.isa_level({"cpu": _CPU, "cpu_flags": "avx2 fma"}) == table.UNRECORDED
