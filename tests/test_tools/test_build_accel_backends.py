# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the compiled Go/Mojo accelerator backend builder

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_tool() -> Any:
    tool_path = _repo_root() / "tools" / "build_accel_backends.py"
    spec = importlib.util.spec_from_file_location("build_accel_backends", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MOD = _load_tool()


def _make_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Create a miniature accel/package tree covering every discovery branch."""
    accel = tmp_path / "accel"
    models = tmp_path / "models"
    # Go sources: conventional (libtheta.so <- theta.go), renamed via recipe
    # (libhr.so <- hindmarsh_rose.go), and an LGSSM-style world_model backend
    # whose loader uses a pathlib `/` expression rather than os.path.join.
    (accel / "go" / "neurons" / "theta").mkdir(parents=True)
    (accel / "go" / "neurons" / "theta" / "theta.go").write_text("package main\n")
    (accel / "go" / "neurons" / "hindmarsh_rose").mkdir(parents=True)
    (accel / "go" / "neurons" / "hindmarsh_rose" / "hindmarsh_rose.go").write_text("package main\n")
    (accel / "go" / "lgssm").mkdir(parents=True)
    (accel / "go" / "lgssm" / "lgssm.go").write_text("package main\n")
    # A loader output whose source is absent -> must be skipped.
    (accel / "go" / "neurons" / "ghost").mkdir(parents=True)
    # Mojo sources: conventional kernel + a renamed one whose recipe lives only in
    # the .mojo header comment (not any .py), plus the LGSSM world_model source.
    (accel / "mojo" / "kernels").mkdir(parents=True)
    (accel / "mojo" / "kernels" / "theta.mojo").write_text("fn main():\n    pass\n")
    (accel / "mojo" / "neurons").mkdir(parents=True)
    (accel / "mojo" / "neurons" / "hindmarsh_rose.mojo").write_text(
        "# mojo build --emit shared-lib -o libhr.so hindmarsh_rose.mojo\nfn main():\n    pass\n"
    )
    (accel / "mojo" / "world_model").mkdir(parents=True)
    (accel / "mojo" / "world_model" / "lgssm.mojo").write_text("fn main():\n    pass\n")
    # A pruned vendored dir must not be scanned (would crash on non-utf8 / noise).
    (accel / "mojo" / ".pixi").mkdir(parents=True)
    (accel / "mojo" / ".pixi" / "poison.py").write_text("this is not valid python !!!\n")
    models.mkdir()
    (models / "theta.py").write_text(
        "import os\n"
        '_ACCEL_ROOT = "x"\n'
        "def ensure_go_loaded():\n"
        '    p = os.path.join(_ACCEL_ROOT, "go", "neurons", "theta", "libtheta.so")\n'
        "def ensure_mojo_loaded():\n"
        '    q = os.path.join(_ACCEL_ROOT, "mojo", "kernels", "libtheta.so")\n'
        "def unrelated():\n"
        '    return os.path.join("no", "root", "here.so")\n'
    )
    (models / "hindmarsh_rose.py").write_text(
        "import os\n"
        '_ACCEL_ROOT = "x"\n'
        "def ensure_go_loaded():\n"
        '    p = os.path.join(_ACCEL_ROOT, "go", "neurons", "hindmarsh_rose", "libhr.so")\n'
        "def ensure_mojo_loaded():\n"
        '    m = os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libhr.so")\n'
        "# build via: go build -buildmode=c-shared -o libhr.so hindmarsh_rose.go\n"
    )
    # LGSSM-style loader outside neurons/models, using pathlib rooted at PACKAGE_ROOT.
    world = tmp_path / "world_model"
    world.mkdir()
    (world / "_lgssm.py").write_text(
        "from pathlib import Path\n"
        "_PACKAGE_ROOT = Path('x')\n"
        "def _ensure_go_loaded():\n"
        '    p = _PACKAGE_ROOT / "accel" / "go" / "lgssm" / "liblgssm.so"\n'
        "def _ensure_mojo_loaded():\n"
        '    m = _PACKAGE_ROOT / "accel" / "mojo" / "world_model" / "liblgssm.so"\n'
    )
    (models / "ghost.py").write_text(
        "import os\n"
        '_ACCEL_ROOT = "x"\n'
        "def ensure_go_loaded():\n"
        '    p = os.path.join(_ACCEL_ROOT, "go", "neurons", "ghost", "libghost.so")\n'
    )
    return accel, models


def _target(tmp_path: Path, language: str = "go") -> Any:
    src = tmp_path / "src.go"
    src.write_text("package main\n")
    return MOD.BackendTarget(language=language, name="x", source=src, output=tmp_path / "libx.so")


# ---- discovery -------------------------------------------------------------


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


# ---- command construction --------------------------------------------------


def test_go_command(tmp_path: Path) -> None:
    target = _target(tmp_path, "go")
    assert MOD._go_command(target) == [
        "go",
        "build",
        "-buildmode=c-shared",
        "-o",
        "libx.so",
        ".",
    ]


def test_mojo_command(tmp_path: Path) -> None:
    target = _target(tmp_path, "mojo")
    cmd = MOD._mojo_command(target, ["pixi", "run", "mojo"])
    assert cmd == [
        "pixi",
        "run",
        "mojo",
        "build",
        "--emit",
        "shared-lib",
        "-o",
        "libx.so",
        "src.go",
    ]


# ---- build_target ----------------------------------------------------------


def _fake_completed(returncode: int, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["x"], returncode=returncode, stdout="", stderr=stderr)


def test_build_target_success_writes_library(tmp_path: Path) -> None:
    target = _target(tmp_path, "go")

    def runner(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        target.output.write_bytes(b"\x7fELF")
        return _fake_completed(0)

    result = MOD.build_target(target, runner=runner)
    assert result.ok and result.detail == "CGO_ENABLED=1"


def test_build_target_mojo_success(tmp_path: Path) -> None:
    target = _target(tmp_path, "mojo")

    def runner(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        target.output.write_bytes(b"\x7fELF")
        return _fake_completed(0)

    result = MOD.build_target(target, mojo_command=["mojo"], runner=runner)
    assert result.ok and result.detail == "ok"


def test_build_target_nonzero_exit(tmp_path: Path) -> None:
    target = _target(tmp_path, "go")
    result = MOD.build_target(target, runner=lambda c, w: _fake_completed(2, "boom\nkaboom"))
    assert not result.ok and "exit 2" in result.detail and "kaboom" in result.detail


def test_build_target_success_but_no_library(tmp_path: Path) -> None:
    target = _target(tmp_path, "go")
    result = MOD.build_target(target, runner=lambda c, w: _fake_completed(0))
    assert not result.ok and "no library" in result.detail


def test_build_target_toolchain_missing(tmp_path: Path) -> None:
    target = _target(tmp_path, "mojo")

    def runner(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("mojo")

    result = MOD.build_target(target, runner=runner)
    assert not result.ok and "toolchain missing" in result.detail


def test_default_runner_sets_cgo_for_go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    # Control the ambient toolchain env so the assertions are deterministic.
    monkeypatch.delenv("GOTOOLCHAIN", raising=False)

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return _fake_completed(0)

    monkeypatch.setattr(MOD.subprocess, "run", fake_run)
    MOD._default_runner(["go", "build"], tmp_path)
    assert captured["env"]["CGO_ENABLED"] == "1"
    # Toolchain auto-management so the newer go.mod requirement is fetched
    # instead of failing under a GOTOOLCHAIN=local CI runner.
    assert captured["env"]["GOTOOLCHAIN"] == "auto"
    # non-go command leaves CGO/toolchain untouched (delegates to inherited env)
    captured.clear()
    monkeypatch.setattr(MOD.subprocess, "run", fake_run)
    MOD._default_runner(["mojo", "build"], tmp_path)
    assert "cmd" in captured
    assert "GOTOOLCHAIN" not in captured["env"]


# ---- main ------------------------------------------------------------------


def _stub_targets(mod: Any, names: list[str]) -> list[Any]:
    return [
        mod.BackendTarget(
            language="go", name=n, source=Path(f"/{n}.go"), output=Path(f"/lib{n}.so")
        )
        for n in names
    ]


def test_main_all_ok(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(MOD, "discover_targets", lambda lang, **k: _stub_targets(MOD, ["theta"]))
    monkeypatch.setattr(MOD, "build_target", lambda t, **k: MOD.BuildResult(t, True, "ok"))
    assert MOD.main(["--language", "go"]) == 0
    assert "built 1/1" in capsys.readouterr().out


def test_main_required_failure_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MOD, "discover_targets", lambda lang, **k: _stub_targets(MOD, ["theta"]))
    monkeypatch.setattr(MOD, "build_target", lambda t, **k: MOD.BuildResult(t, False, "exit 1"))
    assert MOD.main(["--language", "go", "--require", "theta"]) == 1


def test_main_required_never_discovered_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MOD, "discover_targets", lambda lang, **k: _stub_targets(MOD, ["theta"]))
    monkeypatch.setattr(MOD, "build_target", lambda t, **k: MOD.BuildResult(t, True, "ok"))
    assert MOD.main(["--language", "go", "--require", "adex"]) == 1


def test_main_all_languages_and_mojo_command(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_discover(language: str, **kwargs: Any) -> list[Any]:
        seen.append(language)
        return _stub_targets(MOD, [f"{language}_only"])

    monkeypatch.setattr(MOD, "discover_targets", fake_discover)
    monkeypatch.setattr(MOD, "build_target", lambda t, **k: MOD.BuildResult(t, True, "ok"))
    assert MOD.main(["--language", "all", "--mojo-command", "pixi run mojo"]) == 0
    assert seen == ["go", "mojo"]


def test_main_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["build_accel_backends.py", "--language", "go"])
    monkeypatch.setattr(MOD, "discover_targets", lambda lang, **k: [])
    assert MOD.main() == 0
