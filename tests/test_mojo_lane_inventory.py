# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo lane inventory

"""Hold the Mojo lane inventory to its rules, to the repository and to the package's own declarations."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.mojo_lane_inventory import (
    DEFAULT_OUTPUT,
    MODEL_CLASSES,
    build,
    classify,
    code_lines,
    declarations,
    library_literals,
    main,
    models,
    render,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

STUB = """# SC-NeuroCore — Mojo SIMD acceleration for example

fn _compute_grade(verified_pct: Int) -> Int:
    var __compute_grade_line = 'if verified_pct >= 80.0:'
    return 0  # return "A"
    return 0  # return "F"

fn to_dict() -> Int:
    return 0  # return {
    var _to_dict_line = '"total": total,'
"""

FN_IMPLEMENTATION = """from std.memory import UnsafePointer

@always_inline
fn add(a: Int64, b: Int64) -> Int64:
    var total = a + b
    return total
"""

DEF_IMPLEMENTATION = """from std.memory import UnsafePointer

@export
def add(a: Int64, b: Int64) -> Int64:
    return a + b
"""

MULTI_LINE_SIGNATURE = """fn quantise(
    sample: Int64,
) -> Int64:
    return 0
"""

DOCSTRING_ONLY = '''"""A module docstring
that spans lines."""
# and a comment
'''


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (STUB, "translated_stub"),
        (FN_IMPLEMENTATION, "implementation_fn"),
        (DEF_IMPLEMENTATION, "implementation_def"),
        (MULTI_LINE_SIGNATURE, "implementation_fn"),
        (DOCSTRING_ONLY, "comment_only"),
        ("", "comment_only"),
    ],
    ids=["stub", "fn", "def", "multi-line-signature-is-never-a-stub", "docstring-only", "empty"],
)
def test_each_rule_classifies_its_example(text: str, expected: str) -> None:
    """Each class of the module's rules is reached by its own example, and only by it."""
    assert classify(text) == expected


def test_code_lines_skip_comments_and_triple_quoted_blocks() -> None:
    """Comment lines and the inside of a triple-quoted block are not code."""
    text = '# head\n"""open\ninside\nclose"""\nvar x = 1\n  # indented comment\n'
    assert code_lines(text) == ["var x = 1"]


def test_the_committed_inventory_is_current() -> None:
    """The committed inventory equals a fresh build of the repository."""
    committed = (REPO_ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8")
    assert committed == render(build(REPO_ROOT))


def test_every_translated_stub_lies_under_a_declared_mirror_glob() -> None:
    """The package declares its mirrors; no stub may sit outside them as if it were a lane.

    The check runs on a fresh build of the repository, so a new stub outside the mirrors fails here even before the
    committed inventory is regenerated.
    """
    inventory = build(REPO_ROOT)
    stubs = [source for source in inventory["sources"] if source["class"] == "translated_stub"]
    assert stubs
    assert all(source["declared"] == "mirror" for source in stubs)
    assert inventory["counts"]["translated_stub_declared_mirror"] == len(stubs)


def test_counts_add_up() -> None:
    """The classes partition the sources and the models, and attributed literals are a subset of all literals."""
    counts = json.loads((REPO_ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8"))["counts"]
    classes = ("comment_only", "translated_stub", "implementation_fn", "implementation_def")
    assert sum(counts[name] for name in classes) == counts["sources"]
    assert sum(counts[f"models_{name}"] for name in MODEL_CLASSES) == counts["models"] > 0
    assert counts["library_literals_attributed_to_mojo"] <= counts["library_literals"]
    assert counts["implementation_fn_built_by_builder"] <= counts["built_by_builder"]


def write_tree(root: Path) -> None:
    """Write a minimal package tree with declarations, three Mojo sources and a library literal.

    Parameters
    ----------
    root : Path
        The fixture repository root.
    """
    mojo = root / "src" / "sc_neurocore" / "accel" / "mojo"
    (mojo / "kernels").mkdir(parents=True)
    (mojo / "lane").mkdir()
    (mojo / "__init__.py").write_text(
        'AUTHORITATIVE_MOJO_ENTRYPOINTS = ("lane/lane.mojo",)\n'
        'NON_AUTHORITATIVE_MOJO_MIRROR_GLOBS = ("kernels/*.mojo",)\n',
        encoding="utf-8",
    )
    (mojo / "lane" / "lane.mojo").write_text(DEF_IMPLEMENTATION, encoding="utf-8")
    (mojo / "kernels" / "example.mojo").write_text(STUB, encoding="utf-8")
    (mojo / "old.mojo").write_text(FN_IMPLEMENTATION, encoding="utf-8")
    (root / "src" / "sc_neurocore" / "loader.py").write_text(
        'NAME = "liblane.so"\nOTHER = "not a library"\n', encoding="utf-8"
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


def test_a_fixture_tree_is_classified_declared_and_its_literals_listed(tmp_path: Path) -> None:
    """On a small tree: classes, declarations and the literal are found; nothing is attributed without a loader."""
    write_tree(tmp_path)
    track_tree(tmp_path)
    assert declarations(tmp_path) == (("lane/lane.mojo",), ("kernels/*.mojo",))
    assert library_literals(tmp_path) == [("src/sc_neurocore/loader.py", "liblane.so")]
    inventory = build(tmp_path)
    by_path = {
        source["path"].rsplit("accel/mojo/", 1)[1]: source for source in inventory["sources"]
    }
    assert by_path["lane/lane.mojo"]["class"] == "implementation_def"
    assert by_path["lane/lane.mojo"]["declared"] == "authoritative"
    assert by_path["kernels/example.mojo"]["class"] == "translated_stub"
    assert by_path["kernels/example.mojo"]["declared"] == "mirror"
    assert by_path["old.mojo"]["class"] == "implementation_fn"
    assert by_path["old.mojo"]["declared"] is None
    assert inventory["library_literals"] == [
        {
            "python": "src/sc_neurocore/loader.py",
            "library": "liblane.so",
            "builder_mojo_source": None,
        }
    ]


def test_a_tree_without_declarations_declares_nothing(tmp_path: Path) -> None:
    """Without the package's declaration file both declaration lists are empty."""
    assert declarations(tmp_path) == ((), ())


def test_untracked_files_and_ignored_directories_change_nothing(tmp_path: Path) -> None:
    """Only tracked files count: an untracked module and an ignored environment leave the inventory unchanged.

    Both hold a module that cannot be parsed and a library literal, as an installed Mojo environment does; read, either
    would raise or add a literal.
    """
    write_tree(tmp_path)
    track_tree(tmp_path)
    expected = render(build(tmp_path))
    unparsable = 'NAME = "libextra.so"\ndef broken(:\n'
    package = tmp_path / "src" / "sc_neurocore"
    (package / "untracked.py").write_text(unparsable, encoding="utf-8")
    (package / "accel" / "mojo" / "untracked.mojo").write_text(STUB, encoding="utf-8")
    environment = package / "accel" / "mojo" / ".environment"
    environment.mkdir()
    (environment / ".gitignore").write_text("*\n", encoding="utf-8")
    (environment / "vendored.py").write_text(unparsable, encoding="utf-8")
    (environment / "vendored.mojo").write_text(STUB, encoding="utf-8")
    assert render(build(tmp_path)) == expected
    assert library_literals(tmp_path) == [("src/sc_neurocore/loader.py", "liblane.so")]


GHOST_LOADER = (
    "import ctypes\nfrom pathlib import Path\n\n"
    'LIBRARY = ctypes.CDLL(str(Path(__file__).with_name("libghost.so")))\n'
)


def test_a_builder_target_whose_source_is_untracked_is_left_out(tmp_path: Path) -> None:
    """The builder walks the directory; its target counts only when the target's source is tracked.

    A tracked loader names ``libghost.so`` beside it. While ``ghost.mojo`` is untracked, the literal is attributed to
    no source and the inventory equals the one without the file; once it is tracked, the builder's attribution shows.
    """
    write_tree(tmp_path)
    ghost = tmp_path / "src" / "sc_neurocore" / "accel" / "mojo" / "ghost"
    ghost.mkdir()
    (ghost / "loader.py").write_text(GHOST_LOADER, encoding="utf-8")
    track_tree(tmp_path)
    expected = render(build(tmp_path))
    (ghost / "ghost.mojo").write_text(DEF_IMPLEMENTATION, encoding="utf-8")
    assert render(build(tmp_path)) == expected
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    attributed = {
        item["library"]: item["builder_mojo_source"] for item in build(tmp_path)["library_literals"]
    }
    assert attributed["libghost.so"] == "src/sc_neurocore/accel/mojo/ghost/ghost.mojo"


def test_a_root_outside_a_work_tree_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A directory that is not the top of a Git work tree has no tracked files to inventory; it is refused."""
    write_tree(tmp_path)
    with pytest.raises(ValueError, match="not the top of a Git work tree"):
        build(tmp_path)
    assert main(["--root", str(tmp_path), "--check"]) == 2
    assert "not the top of a Git work tree" in capsys.readouterr().err


def test_a_tracked_file_missing_from_the_work_tree_is_refused(tmp_path: Path) -> None:
    """A tracked source deleted from the work tree is named, not silently left out."""
    write_tree(tmp_path)
    track_tree(tmp_path)
    (tmp_path / "src" / "sc_neurocore" / "accel" / "mojo" / "old.mojo").unlink()
    with pytest.raises(ValueError, match="src/sc_neurocore/accel/mojo/old.mojo"):
        build(tmp_path)


def test_the_command_writes_checks_and_summarises(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--check`` fails on an absent or drifted file and passes after ``--write``; ``--summary`` prints the counts."""
    write_tree(tmp_path)
    track_tree(tmp_path)
    output = tmp_path / "inventory.json"
    root = ["--root", str(tmp_path), "--output", str(output)]
    assert main([*root, "--check"]) == 1
    assert main([*root, "--write"]) == 0
    assert main([*root, "--check"]) == 0
    output.write_text(output.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert main([*root, "--check"]) == 1
    capsys.readouterr()
    assert main([*root, "--summary"]) == 0
    assert "translated_stub: 1" in capsys.readouterr().out


MODEL_SOURCES = {
    "accel/backend_order.py": (
        'ACCELERATORS: tuple[str, ...] = ("rust", "mojo", "julia", "go")\n\n\n'
        'def with_floor(floor="python"):\n    return (*ACCELERATORS, floor)\n'
    ),
    "accel/backend_selection.py": "def select_backend_order(kernel, static=()):\n    return static\n",
    "accel/dispatch_x.py": (
        "from sc_neurocore.accel.backend_selection import select_backend_order\n\n\n"
        'def auto_backend():\n    return select_backend_order("k")[0]\n'
    ),
    "neurons/models/m_selector.py": (
        "from sc_neurocore.accel.backend_selection import select_backend_order\n\n\n"
        'def pick(backend="auto"):\n    return select_backend_order("k")[0] if backend == "auto" else backend\n'
    ),
    "neurons/models/m_via.py": (
        'def pick(backend="auto"):\n'
        "    from sc_neurocore.accel import dispatch_x as backends\n\n"
        '    return backends.auto_backend() if backend == "auto" else backend\n'
    ),
    "neurons/models/m_fixed_chain.py": (
        'def pick(backend="auto", has=True):\n'
        '    selected = backend\n    if backend == "auto":\n'
        '        if has:\n            selected = "rust"\n        else:\n            selected = "julia"\n'
        "    return selected\n"
    ),
    "neurons/models/m_fixed_static.py": (
        "from sc_neurocore.accel.backend_order import with_floor\n\n"
        '_ORDER = with_floor("python")\n\n\n'
        "def _first():\n    return _ORDER[0]\n\n\n"
        'def pick(backend="auto"):\n    return _first() if backend == "auto" else backend\n'
    ),
    "neurons/models/m_static_loop.py": (
        "from sc_neurocore.accel.backend_order import ACCELERATORS\n\n\n"
        "def first_available(available):\n"
        "    return next((name for name in ACCELERATORS if name in available), None)\n"
    ),
    "neurons/models/m_rust_only.py": (
        'def pick(backend="auto", has=True):\n'
        '    choices = ("rust", "julia", "go", "mojo")\n'
        '    assert backend == "auto" or backend in choices\n'
        '    return "rust" if backend == "auto" and has else backend\n'
    ),
    "neurons/models/m_rust_test.py": (
        'def pick(backend="auto", has=True):\n'
        '    if backend == "rust" or (backend == "auto" and has):\n        return 1\n    return 0\n'
    ),
    "neurons/models/m_rust_bool.py": (
        "import ctypes\n\n\n"
        'def pick(backend="auto", has=True):\n'
        '    prefer_rust = backend == "rust" or (backend == "auto" and has)\n'
        "    if prefer_rust:\n        return 1\n"
        '    if backend == "mojo":\n        return ctypes.CDLL("libm.so.6")\n'
        "    return 0\n"
    ),
    "neurons/models/m_engine.py": (
        'import importlib\n\nENGINE = importlib.import_module("sc_neurocore_engine")\n'
    ),
    "neurons/models/m_direct_library.py": 'import ctypes\n\nLIB = ctypes.CDLL("libmodel.so")\n',
    "neurons/models/m_direct_package.py": "from sc_neurocore.accel.mojo.loader import load\n",
    "neurons/models/m_plain.py": "def step(v):\n    return v + 1\n",
}


def write_models(root: Path) -> None:
    """Write a package with the backend order, a selector, one dispatch module and one model per rule.

    Parameters
    ----------
    root : Path
        The fixture repository root.
    """
    package = root / "src" / "sc_neurocore"
    for relative, text in MODEL_SOURCES.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        ("m_selector", "selector"),
        ("m_via", "selector"),
        ("m_fixed_chain", "fixed_order"),
        ("m_fixed_static", "fixed_order"),
        ("m_static_loop", "fixed_order"),
        ("m_rust_only", "single_backend"),
        ("m_rust_test", "single_backend"),
        ("m_rust_bool", "single_backend"),
        ("m_engine", "single_backend"),
        ("m_direct_library", "direct_backend"),
        ("m_direct_package", "direct_backend"),
        ("m_plain", "no_accelerated_backend"),
    ],
)
def test_each_model_rule_classifies_its_example(tmp_path: Path, module: str, expected: str) -> None:
    """Each model class is reached by its own example, through the stated precedence."""
    write_models(tmp_path)
    track_tree(tmp_path)
    by_name = {Path(record["path"]).stem: record for record in models(tmp_path)}
    assert set(by_name) == {
        Path(name).stem for name in MODEL_SOURCES if name.startswith("neurons/")
    }
    assert by_name[module]["class"] == expected


def test_a_dispatch_module_is_read_one_hop_and_named(tmp_path: Path) -> None:
    """A model that delegates to an ``accel`` module takes that module's selector call, and the record names it."""
    write_models(tmp_path)
    track_tree(tmp_path)
    via = next(record for record in models(tmp_path) if record["path"].endswith("m_via.py"))
    assert via["dispatch_modules"] == ["src/sc_neurocore/accel/dispatch_x.py"]
    assert via["features"]["calls_selector"] and via["features"]["dispatch_modules"] == [
        "dispatch_x"
    ]


def test_an_auto_choice_outranks_names_used_for_explicit_backends(tmp_path: Path) -> None:
    """Names a model accepts for an explicit backend do not make its ``auto`` a fixed order."""
    write_models(tmp_path)
    track_tree(tmp_path)
    rust_only = next(
        record for record in models(tmp_path) if record["path"].endswith("m_rust_only.py")
    )
    assert rust_only["features"]["backend_names"] == ["go", "julia", "mojo", "rust"]
    assert rust_only["features"]["auto_backend_names"] == ["rust"]
    assert rust_only["class"] == "single_backend"


def test_without_the_backend_order_no_name_is_an_accelerator(tmp_path: Path) -> None:
    """The accelerated backends come from the package's own order; without it, names alone class nothing."""
    write_models(tmp_path)
    (tmp_path / "src" / "sc_neurocore" / "accel" / "backend_order.py").unlink()
    track_tree(tmp_path)
    chain = next(
        record for record in models(tmp_path) if record["path"].endswith("m_fixed_chain.py")
    )
    assert chain["class"] == "no_accelerated_backend"
