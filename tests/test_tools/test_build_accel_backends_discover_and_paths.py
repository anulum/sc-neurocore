# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (discover_and_paths) from former test_build_accel_backends.py

from __future__ import annotations

from build_accel_backends_support import *  # noqa: F403


def test_discover_go_pairs_convention_recipe_and_pathlib(tmp_path: Path) -> None:
    accel, _ = _make_tree(tmp_path)
    targets = MOD.discover_targets("go", accel_root=accel)
    by_name = {t.name: t for t in targets}
    assert set(by_name) == {"theta", "hindmarsh_rose", "lgssm"}  # ghost skipped (no source)
    assert by_name["theta"].output.name == "libtheta.so"
    assert by_name["hindmarsh_rose"].output.name == "libhr.so"  # recipe-paired
    assert by_name["hindmarsh_rose"].source.name == "hindmarsh_rose.go"
    # LGSSM discovered through a pathlib loader rooted at _PACKAGE_ROOT.
    assert by_name["lgssm"].output == accel / "go" / "lgssm" / "liblgssm.so"
    assert by_name["lgssm"].source == accel / "go" / "lgssm" / "lgssm.go"


def test_discover_mojo_uses_convention_recipe_and_pathlib(tmp_path: Path) -> None:
    accel, _ = _make_tree(tmp_path)
    by_name = {t.name: t for t in MOD.discover_targets("mojo", accel_root=accel)}
    assert set(by_name) == {"theta", "hindmarsh_rose", "lgssm"}
    # hindmarsh_rose mojo is paired only by the recipe in its .mojo header comment.
    assert by_name["hindmarsh_rose"].output.name == "libhr.so"
    assert by_name["hindmarsh_rose"].source.name == "hindmarsh_rose.mojo"
    assert by_name["lgssm"].output == accel / "mojo" / "world_model" / "liblgssm.so"


def test_loader_output_paths_ignores_unrooted_joins(tmp_path: Path) -> None:
    accel, models = _make_tree(tmp_path)
    outs = MOD._loader_output_paths("go", sorted(models.glob("*.py")), accel)
    names = {p.name for p in outs}
    assert "libtheta.so" in names and "libhr.so" in names
    assert "here.so" not in names  # the unrooted os.path.join is not a backend


def test_loader_output_paths_skips_unparsable(tmp_path: Path) -> None:
    bad = tmp_path / "broken.py"
    bad.write_text("def f(:\n")  # SyntaxError
    assert MOD._loader_output_paths("go", [bad], tmp_path) == set()


def test_iter_files_prunes_vendored_dirs(tmp_path: Path) -> None:
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep" / "a.py").write_text("x = 1\n")
    (tmp_path / ".pixi" / "deep").mkdir(parents=True)
    (tmp_path / ".pixi" / "deep" / "skip.py").write_text("y = 2\n")
    found = {p.name for p in MOD._iter_files(tmp_path, ".py")}
    assert found == {"a.py"}


def test_loader_lib_parts_matches_and_rejects() -> None:
    import ast

    def expr(src: str) -> ast.AST:
        return ast.parse(src, mode="eval").body

    # os.path.join rooted at *_ROOT
    assert MOD._loader_lib_parts(expr('os.path.join(_ACCEL_ROOT, "go", "x.so")')) == (
        "_ACCEL_ROOT",
        ["go", "x.so"],
    )
    # pathlib chain rooted at *_ROOT
    assert MOD._loader_lib_parts(expr('_PACKAGE_ROOT / "accel" / "go" / "l.so"')) == (
        "_PACKAGE_ROOT",
        ["accel", "go", "l.so"],
    )
    # not a join / not a division -> None
    assert MOD._loader_lib_parts(expr("os.path.dirname(x)")) is None
    # join without a *_ROOT anchor -> None
    assert MOD._loader_lib_parts(expr('os.path.join("a", "b")')) is None
    # join with a *_ROOT anchor but no string parts -> None
    assert MOD._loader_lib_parts(expr("os.path.join(_ACCEL_ROOT, other)")) is None
    # pathlib chain whose right operand is not a string constant -> None
    assert MOD._loader_lib_parts(expr("_PACKAGE_ROOT / other / x")) is None
    # pathlib chain not rooted at a *_ROOT name -> None
    assert MOD._loader_lib_parts(expr('base / "accel" / "x.so"')) is None


def test_conventional_source_name() -> None:
    assert MOD._conventional_source_name("libcoba_lif.so", ".go") == "coba_lif.go"
    assert MOD._conventional_source_name("plain.so", ".mojo") == "plain.mojo"


def test_hint_pairs_and_read_text(tmp_path: Path) -> None:
    good = tmp_path / "a.py"
    good.write_text("go build -buildmode=c-shared -o libhr.so hindmarsh_rose.go\n")
    missing = tmp_path / "does_not_exist.py"
    text = MOD._read_text([good, missing])
    pairs = MOD._hint_pairs(text, MOD._GO_HINT)
    assert pairs == {"hindmarsh_rose.go": "libhr.so"}
