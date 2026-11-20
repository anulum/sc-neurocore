# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia lane inventory

"""Hold the Julia lane inventory to its rules, to the repository and to the package's own declarations."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from tools.julia_lane_inventory import (
    CLASSES,
    TRANSCRIPTION_RULES,
    DEFAULT_OUTPUT,
    Inventory,
    JuliaUnavailable,
    build,
    carried_verdicts,
    classify,
    declarations,
    main,
    parse_with_julia,
    references,
    render,
    transcription,
)
from tools.mojo_lane_inventory import tracked

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Set to 1 on a gate host that has Julia, so that its absence is a failure there.
REQUIRE_JULIA = os.environ.get("SC_NEUROCORE_REQUIRE_JULIA_PARSE") == "1"

JULIA_KERNEL = "module Kernel\nadd(a, b) = a + b\nend\n"
TRANSCRIPT = 'function f(xs)\n    out = push!(, xs)\n    return f"{xs}"\nend\n'


def fake_parser(root: Path, paths: Sequence[str]) -> tuple[str, dict[str, bool]]:
    """Parse by a stand-in rule: a file parses unless it holds the transcript's ``push!(, `` form."""
    return "fixture", {
        path: "push!(, " not in (root / path).read_text(encoding="utf-8") for path in paths
    }


def absent_parser(root: Path, paths: Sequence[str]) -> tuple[str, dict[str, bool]]:
    """Behave as a host without Julia."""
    raise JuliaUnavailable("no julia program on PATH; parse verdicts cannot be computed here")


def write_tree(root: Path) -> None:
    """Write a package whose Julia files are reached in every way the rules know, and in none.

    Parameters
    ----------
    root : Path
        The fixture repository root.
    """
    package = root / "src" / "sc_neurocore"
    julia = package / "accel" / "julia"
    for directory in ("lane", "mirrors", "pkg", "analysis"):
        (julia / directory).mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "accel" / "__init__.py").write_text("", encoding="utf-8")
    (julia / "__init__.py").write_text(
        'AUTHORITATIVE_JULIA_ENTRYPOINTS = ("lane/kernel.jl",)\n'
        'NON_AUTHORITATIVE_JULIA_MIRROR_GLOBS = ("mirrors/*.jl", "analysis/*.jl")\n',
        encoding="utf-8",
    )
    (julia / "lane" / "kernel.jl").write_text(JULIA_KERNEL, encoding="utf-8")
    (julia / "joined.jl").write_text(JULIA_KERNEL, encoding="utf-8")
    (julia / "helper.jl").write_text(JULIA_KERNEL, encoding="utf-8")
    (julia / "analysis" / "loaded_mirror.jl").write_text(JULIA_KERNEL, encoding="utf-8")
    (julia / "mirrors" / "stub.jl").write_text(TRANSCRIPT, encoding="utf-8")
    (julia / "undeclared.jl").write_text(TRANSCRIPT, encoding="utf-8")
    (julia / "pkg" / "kernel.jl").write_text(JULIA_KERNEL, encoding="utf-8")
    (julia / "pkg" / "_runtime.py").write_text(
        "from pathlib import Path\n\nKERNEL_DIR = Path(__file__).resolve().parent\n",
        encoding="utf-8",
    )
    (julia / "pkg" / "__init__.py").write_text(
        "from . import _runtime\n\n_KERNEL_DIR = _runtime.KERNEL_DIR\n\n\n"
        "def load(jl):\n"
        '    jl_path = _KERNEL_DIR / "kernel.jl"\n'
        "    jl.include(str(jl_path))\n",
        encoding="utf-8",
    )
    (package / "loaders.py").write_text(
        "import os as _os\n"
        "from pathlib import Path\n\n"
        '_ACCEL_ROOT = _os.path.join(_os.path.dirname(__file__), "accel")\n\n\n'
        "def _accel_path(*parts):\n"
        '    """Return a path under the accel tree."""\n'
        "    root = _os.path.dirname(__file__)\n"
        '    return _os.path.join(root, "accel", *parts)\n\n\n'
        "def load(jl):\n"
        '    jl.include(_os.path.abspath(_os.path.join(_ACCEL_ROOT, "julia", "joined.jl")))\n'
        '    jl.include(_accel_path("julia", "helper.jl"))\n'
        '    jl.include(str(Path(__file__).parent / "accel/julia/analysis/loaded_mirror.jl"))\n'
        '    jl.include(_os.path.join(_ACCEL_ROOT, "julia", "absent.jl"))\n'
        '    names = ["kernel.jl", "other.jl"]\n'
        "    return names\n",
        encoding="utf-8",
    )
    (package / "dynamic_names.py").write_text(
        "from pathlib import Path\n\n\n"
        "def paths(name, stem):\n"
        '    first = f"{name}.jl"\n'
        '    second = stem + ".jl"\n'
        '    third = Path(name).with_suffix(".jl")\n'
        "    return first, second, third\n",
        encoding="utf-8",
    )
    (julia / "mirrors" / "deeper").mkdir()
    (julia / "mirrors" / "deeper" / "nested.jl").write_text(JULIA_KERNEL, encoding="utf-8")
    benchmarks = root / "benchmarks"
    benchmarks.mkdir()
    (benchmarks / "bench_lane.py").write_text(
        'SOURCES = [\n    "src/sc_neurocore/accel/julia/lane/kernel.jl",\n]\n'
        'SUFFIXES = {".jl", ".py"}\nNOTE = "Aqua.jl and friends"\n',
        encoding="utf-8",
    )


def track_tree(root: Path) -> None:
    """Make ``root`` a Git work tree and track every file in it.

    Parameters
    ----------
    root : Path
        The fixture repository root.
    """
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)


def fixture_inventory(root: Path) -> Inventory:
    """Build the fixture's inventory with the stand-in parser."""
    paths = [path.relative_to(root).as_posix() for path in tracked(root, "src/sc_neurocore", ".jl")]
    version, verdicts = fake_parser(root, paths)
    return build(root, verdicts, version)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ((True, True, True), "authoritative_entrypoint"),
        ((False, True, True), "loaded_by_python"),
        ((False, False, True), "declared_mirror"),
        ((False, False, False), "undeclared"),
    ],
)
def test_the_class_follows_the_stated_precedence(
    flags: tuple[bool, bool, bool], expected: str
) -> None:
    """Authoritative, then loaded, then mirror, then undeclared."""
    assert classify(*flags) == expected
    assert expected in CLASSES


def test_a_fixture_tree_is_classified_with_every_reference_resolved_or_listed(
    tmp_path: Path,
) -> None:
    """Each way of building a path resolves to its file; names in lists stay listed as unresolved."""
    write_tree(tmp_path)
    track_tree(tmp_path)
    assert declarations(tmp_path) == (("lane/kernel.jl",), ("mirrors/*.jl", "analysis/*.jl"))
    inventory = fixture_inventory(tmp_path)
    by_path = {
        source["path"].rsplit("accel/julia/", 1)[1]: source for source in inventory["sources"]
    }
    assert by_path["lane/kernel.jl"]["class"] == "authoritative_entrypoint"
    assert by_path["joined.jl"]["class"] == "loaded_by_python"
    assert by_path["helper.jl"]["class"] == "loaded_by_python"
    assert by_path["pkg/kernel.jl"]["loaders"] == ["src/sc_neurocore/accel/julia/pkg/__init__.py"]
    assert by_path["mirrors/stub.jl"]["class"] == "declared_mirror"
    assert not by_path["mirrors/stub.jl"]["parses"]
    assert by_path["undeclared.jl"]["class"] == "undeclared"
    loaded_mirror = by_path["analysis/loaded_mirror.jl"]
    assert (loaded_mirror["loaded"], loaded_mirror["mirror"], loaded_mirror["class"]) == (
        True,
        True,
        "loaded_by_python",
    )
    assert inventory["counts"]["loaded_under_a_mirror_glob"] == 1
    statuses = {
        (item["python"].rsplit("/", 1)[1], item["status"], item["target"])
        for item in inventory["references"]
    }
    assert ("loaders.py", "missing", "src/sc_neurocore/accel/julia/absent.jl") in statuses
    assert ("bench_lane.py", "named", "src/sc_neurocore/accel/julia/lane/kernel.jl") in statuses
    unresolved = [
        (item["python"].rsplit("/", 1)[1], item["line"])
        for item in inventory["references"]
        if item["status"] == "unresolved"
    ]
    assert unresolved == [
        ("dynamic_names.py", 5),
        ("dynamic_names.py", 6),
        ("dynamic_names.py", 7),
        ("loaders.py", 18),
        ("loaders.py", 18),
    ]
    assert by_path["mirrors/deeper/nested.jl"]["mirror"] is False
    assert by_path["mirrors/deeper/nested.jl"]["class"] == "undeclared"
    assert inventory["counts"]["references"] == sum(
        inventory["counts"][f"references_{status}"]
        for status in ("resolved", "named", "missing", "unresolved")
    )


def test_counts_add_up(tmp_path: Path) -> None:
    """The classes partition the sources, and parse counts add up."""
    write_tree(tmp_path)
    track_tree(tmp_path)
    counts = fixture_inventory(tmp_path)["counts"]
    assert sum(counts[name] for name in CLASSES) == counts["sources"] == 8
    assert counts["parses"] + counts["does_not_parse"] == counts["sources"]
    assert counts["does_not_parse"] == 2


def test_untracked_files_and_ignored_directories_change_nothing(tmp_path: Path) -> None:
    """Only tracked files count: an untracked loader and an ignored environment leave the inventory unchanged."""
    write_tree(tmp_path)
    track_tree(tmp_path)
    expected = render(fixture_inventory(tmp_path))
    package = tmp_path / "src" / "sc_neurocore"
    (package / "untracked.py").write_text('PATH = "untracked.jl"\ndef broken(:\n', encoding="utf-8")
    (package / "accel" / "julia" / "untracked.jl").write_text(TRANSCRIPT, encoding="utf-8")
    environment = package / "accel" / "julia" / ".environment"
    environment.mkdir()
    (environment / ".gitignore").write_text("*\n", encoding="utf-8")
    (environment / "vendored.jl").write_text(TRANSCRIPT, encoding="utf-8")
    tracked_paths = [source["path"] for source in json.loads(expected)["sources"]]
    version, verdicts = fake_parser(tmp_path, tracked_paths)
    assert render(build(tmp_path, verdicts, version)) == expected


def test_verdicts_that_miss_a_tracked_file_are_refused(tmp_path: Path) -> None:
    """A verdict set that does not cover every tracked file cannot build an inventory."""
    write_tree(tmp_path)
    track_tree(tmp_path)
    with pytest.raises(ValueError, match="do not cover exactly"):
        build(tmp_path, {}, "fixture")


def test_a_root_outside_a_work_tree_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A directory that is not the top of a Git work tree has no tracked files to inventory."""
    write_tree(tmp_path)
    assert main(["--root", str(tmp_path), "--check"], parser=fake_parser) == 2
    assert "not the top of a Git work tree" in capsys.readouterr().err


def test_a_tree_without_python_modules_is_refused(tmp_path: Path) -> None:
    """Loader resolution fails closed: no tracked Python module is a refusal, not an empty list."""
    julia = tmp_path / "src" / "sc_neurocore" / "accel" / "julia"
    julia.mkdir(parents=True)
    (julia / "kernel.jl").write_text(JULIA_KERNEL, encoding="utf-8")
    track_tree(tmp_path)
    with pytest.raises(ValueError, match="no tracked Python module"):
        references(tmp_path, {"src/sc_neurocore/accel/julia/kernel.jl"})


def test_a_tracked_file_missing_from_the_work_tree_is_refused(tmp_path: Path) -> None:
    """A tracked source deleted from the work tree is named, not silently left out."""
    write_tree(tmp_path)
    track_tree(tmp_path)
    (tmp_path / "src" / "sc_neurocore" / "accel" / "julia" / "undeclared.jl").unlink()
    with pytest.raises(ValueError, match="undeclared.jl"):
        fixture_inventory(tmp_path)


def test_the_command_writes_checks_carries_verdicts_and_refuses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--check`` re-parses with Julia or carries unchanged verdicts without it; a changed file needs Julia."""
    write_tree(tmp_path)
    track_tree(tmp_path)
    output = tmp_path / "inventory.json"
    root = ["--root", str(tmp_path), "--output", str(output)]
    assert main([*root, "--check"], parser=fake_parser) == 1
    assert main([*root, "--write"], parser=absent_parser) == 2
    assert main([*root, "--write"], parser=fake_parser) == 0
    assert main([*root, "--check"], parser=fake_parser) == 0
    assert "re-parsed every file with julia fixture" in capsys.readouterr().out
    assert main([*root, "--check"], parser=absent_parser) == 0
    assert "parse verdicts carried by sha256; julia absent" in capsys.readouterr().out
    assert main([*root, "--check", "--require-julia"], parser=absent_parser) == 2
    kernel = tmp_path / "src" / "sc_neurocore" / "accel" / "julia" / "helper.jl"
    kernel.write_text(TRANSCRIPT, encoding="utf-8")
    assert main([*root, "--check"], parser=absent_parser) == 1
    assert "changed since the record" in capsys.readouterr().err
    assert main([*root, "--check"], parser=fake_parser) == 1
    assert main([*root, "--write"], parser=fake_parser) == 0
    record: Inventory = json.loads(output.read_text(encoding="utf-8"))
    verdicts, changed = carried_verdicts(tmp_path, record)
    assert changed == [] and verdicts["src/sc_neurocore/accel/julia/helper.jl"] is False
    output.write_text(output.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert main([*root, "--check"], parser=fake_parser) == 1
    capsys.readouterr()
    assert main([*root, "--summary"], parser=fake_parser) == 0
    assert "undeclared: 2" in capsys.readouterr().out


def test_julia_parses_a_kernel_and_refuses_a_transcript(tmp_path: Path) -> None:
    """With Julia present, the real parser accepts Julia and refuses a transcript; absent, it says so.

    On a host where ``SC_NEUROCORE_REQUIRE_JULIA_PARSE=1``, an absent Julia fails this test.
    """
    (tmp_path / "kernel.jl").write_text(JULIA_KERNEL, encoding="utf-8")
    (tmp_path / "transcript.jl").write_text(TRANSCRIPT, encoding="utf-8")
    (tmp_path / "unicode.jl").write_text('# Šotek — ü\nx = "ä" * "ő"\n', encoding="utf-8")
    paths = ["kernel.jl", "transcript.jl", "unicode.jl"]
    if shutil.which("julia") is None:
        assert not REQUIRE_JULIA, "Julia is required on this host and absent"
        with pytest.raises(JuliaUnavailable, match="no julia program"):
            parse_with_julia(tmp_path, paths)
        return
    version, verdicts = parse_with_julia(tmp_path, paths)
    assert version
    assert verdicts == {"kernel.jl": True, "transcript.jl": False, "unicode.jl": True}


def test_the_committed_inventory_is_current(capsys: pytest.CaptureFixture[str]) -> None:
    """The committed inventory equals a fresh build: re-parsed where Julia is, carried by sha256 where it is not."""
    arguments = ["--check", *(["--require-julia"] if REQUIRE_JULIA else [])]
    assert main(arguments) == 0, capsys.readouterr().err
    assert "julia lane inventory check:" in capsys.readouterr().out


def test_every_authoritative_or_loaded_file_parses() -> None:
    """A Julia file the package declares authoritative or a Python module loads must be Julia.

    The verdicts are the committed record's, which the previous test holds current.
    """
    record: Inventory = json.loads((REPO_ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    broken = [
        source["path"]
        for source in record["sources"]
        if (source["authoritative"] or source["loaded"]) and not source["parses"]
    ]
    assert broken == []
    assert record["counts"]["authoritative_entrypoint"] > 0
    assert record["counts"]["loaded_by_python"] > 0


MODULES = {
    "src/sc_neurocore/chip_compiler/compiler.py",
    "src/sc_neurocore/neurons/__init__.py",
    "src/sc_neurocore/adapters/holonomic/_jax_compat.py",
    "src/sc_neurocore/a/b_c/d.py",
    "src/sc_neurocore/a_b/c/d.py",
}
SOURCE = {
    "src/sc_neurocore/chip_compiler/compiler.py": {
        "bits = chip.core.weight_bits",
        "quantized = []",
    },
}


def _header(module: str) -> str:
    """Return a transcript header naming ``module``."""
    return f"# SC-NeuroCore — Julia acceleration for {module}\n"


def _read(module: str) -> set[str]:
    """Return a module's stripped lines from the fixture."""
    return SOURCE.get(module, set())


@pytest.mark.parametrize(
    ("text", "module", "rule", "candidates"),
    [
        (
            _header("chip_compiler/compiler"),
            "src/sc_neurocore/chip_compiler/compiler.py",
            "header",
            [],
        ),
        (_header("neurons"), "src/sc_neurocore/neurons/__init__.py", "header", []),
        (
            _header("adapters_holonomic/_jax_compat"),
            "src/sc_neurocore/adapters/holonomic/_jax_compat.py",
            "header_underscore_dir",
            [],
        ),
        (
            _header("a_b_c/d"),
            None,
            "ambiguous",
            ["src/sc_neurocore/a/b_c/d.py", "src/sc_neurocore/a_b/c/d.py"],
        ),
        (_header("missing/module"), None, "none", []),
        ("x = push!(, xs)\n", None, "none", []),
    ],
    ids=["header", "header-package", "underscore-dir", "ambiguous", "missing-module", "no-header"],
)
def test_each_transcription_rule_names_its_module(
    text: str, module: str | None, rule: str, candidates: list[str]
) -> None:
    """Each rule names the module it should, and an ambiguous name lists its candidates without choosing."""
    found = transcription(text, MODULES, _read)
    assert (found["transcribed_module"], found["transcription_rule"]) == (module, rule)
    assert found["transcription_candidates"] == candidates
    assert rule in TRANSCRIPTION_RULES


def test_the_line_share_counts_non_comment_lines_found_in_the_module() -> None:
    """Of the transcript's non-empty, non-comment lines, those found stripped in the module are counted."""
    text = (
        _header("chip_compiler/compiler")
        + "    bits = chip.core.weight_bits\n\n"
        + "    quantized = []\n"
        + "    weights = push!(, q)\n"
        + "# a comment that is not counted\n"
    )
    found = transcription(text, MODULES, _read)
    assert found["verbatim_line_share"] == "2/3"


def test_every_transcript_of_the_repository_names_its_module() -> None:
    """In the committed record, every file Julia refuses has a rule, and every file it parses has none."""
    record: Inventory = json.loads((REPO_ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    counts = record["counts"]
    for source in record["sources"]:
        if source["parses"]:
            assert source["transcription_rule"] == "not_a_transcript"
        else:
            assert source["transcription_rule"] in TRANSCRIPTION_RULES
            if source["transcription_rule"] in ("header", "header_underscore_dir"):
                assert source["transcribed_module"] is not None
                assert source["verbatim_line_share"] is not None
    assert (
        sum(counts[f"transcription_rule_{rule}"] for rule in TRANSCRIPTION_RULES)
        == counts["does_not_parse"]
    )
    assert counts["transcription_rule_not_a_transcript"] == counts["parses"]
