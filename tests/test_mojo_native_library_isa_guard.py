# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — the instruction-set guard on every Mojo library load

"""A host below the Mojo build level never calls ``ctypes.CDLL`` on a Mojo library, and its lanes fall to the floor."""

from __future__ import annotations

import ast
import ctypes
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sc_neurocore.accel import event_recordings, host_isa
from sc_neurocore.accel.mojo import native_library
from sc_neurocore.accel.mojo.isa_baseline import MOJO_TARGET_CPU
from sc_neurocore.accel.mojo.native_library import (
    MojoIsaRefusal,
    load_mojo_library,
    refusal_for_host,
)
from sc_neurocore.neurons.models import chialvo_map
from tools.benchmark_evidence_gate import tracked_files

SRC = Path(__file__).resolve().parents[1] / "src" / "sc_neurocore"


def _flags_up_to(level: str) -> str:
    """Return a ``/proc/cpuinfo`` text whose flags reach exactly ``level``."""
    names: set[str] = set()
    for each in host_isa.LEVELS[: host_isa.LEVELS.index(level) + 1]:
        names |= host_isa.LEVEL_FLAGS[each]
    return "processor\t: 0\nflags\t\t: " + " ".join(sorted(names)) + "\n"


class _Loads:
    """Record every ``ctypes.CDLL`` call instead of loading a library."""

    def __init__(self) -> None:
        self.paths: list[str] = []

    def __call__(self, path: str, *args: Any, **kwargs: Any) -> object:
        """Record the path and return a stand-in library."""
        self.paths.append(path)
        return object()


@pytest.fixture()
def loads(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Loads]:
    """Replace ``ctypes.CDLL`` with a recorder for the test."""
    recorder = _Loads()
    monkeypatch.setattr(ctypes, "CDLL", recorder)
    yield recorder


def _host_is(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str, machine: str = "x86_64"
) -> None:
    """Make the guard read ``text`` as this host's ``/proc/cpuinfo``."""
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(text, encoding="utf-8")
    monkeypatch.setattr(
        native_library,
        "host_level",
        lambda: host_isa.host_level(cpuinfo=cpuinfo, machine=machine, system="Linux"),
    )
    native_library._host.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_host_cache() -> Iterator[None]:
    """Read the host level afresh in every test, and leave no injected level behind."""
    native_library._host.cache_clear()
    yield
    native_library._host.cache_clear()


def test_the_build_level_is_a_psabi_level() -> None:
    """The level every Mojo library is built for is one the host reader names."""
    assert MOJO_TARGET_CPU in host_isa.LEVELS


def test_a_v2_host_refuses_before_any_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, loads: _Loads
) -> None:
    """On an x86-64-v2 host the refusal is typed, is an OSError, and ``ctypes.CDLL`` is never called."""
    _host_is(monkeypatch, tmp_path, _flags_up_to("x86-64-v2"))
    with pytest.raises(MojoIsaRefusal) as refused:
        load_mojo_library(tmp_path / "libkernel.so")
    assert isinstance(refused.value, OSError)
    assert (refused.value.host, refused.value.required) == ("x86-64-v2", MOJO_TARGET_CPU)
    assert refused.value.library == str(tmp_path / "libkernel.so")
    assert loads.paths == []
    assert isinstance(refusal_for_host(), MojoIsaRefusal)


@pytest.mark.parametrize("level", ["x86-64-v3", "x86-64-v4"])
def test_a_host_at_or_above_the_build_level_loads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, loads: _Loads, level: str
) -> None:
    """At the build level or above, the library is loaded once, by its path."""
    _host_is(monkeypatch, tmp_path, _flags_up_to(level))
    load_mojo_library(tmp_path / "libkernel.so")
    assert loads.paths == [str(tmp_path / "libkernel.so")]
    assert refusal_for_host() is None


def test_an_unknown_host_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, loads: _Loads
) -> None:
    """A host that is not x86-64 Linux gets no guessed level, so no Mojo library is loaded."""
    _host_is(monkeypatch, tmp_path, _flags_up_to("x86-64-v4"), machine="aarch64")
    with pytest.raises(MojoIsaRefusal, match="this host is unknown"):
        load_mojo_library("libkernel.so")
    assert loads.paths == []


def test_a_shared_loader_refuses_only_its_mojo_lane(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, loads: _Loads
) -> None:
    """The Go and Mojo loader of a model refuses the Mojo library on a v2 host and still loads the Go one."""
    _host_is(monkeypatch, tmp_path, _flags_up_to("x86-64-v2"))
    library = tmp_path / "libchialvo.so"
    library.write_bytes(b"")
    assert chialvo_map._load_c_backend(str(library), "chialvo_map_simulate_c", mojo=True) is None
    assert loads.paths == []
    chialvo_map._load_c_backend(str(library), "chialvo_map_simulate_c", mojo=False)
    assert loads.paths == [str(library)]


def test_auto_decoding_falls_to_the_floor_and_an_explicit_request_names_the_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, loads: _Loads
) -> None:
    """With the Mojo lane refused, "auto" decodes on the NumPy floor; asking for Mojo raises with the refusal as cause."""
    _host_is(monkeypatch, tmp_path, _flags_up_to("x86-64-v2"))
    library = tmp_path / "libnmnist.so"
    library.write_bytes(b"")
    monkeypatch.setenv("SC_NEUROCORE_DATASET_MOJO_LIBRARY", str(library))
    monkeypatch.setattr(event_recordings, "select_backend_order", lambda *a, **k: ("mojo", "numpy"))
    event_recordings._load_native_library.cache_clear()
    raw = bytes([3, 4, 0x80, 0x00, 0x10])
    decoded = event_recordings.decode_nmnist_recording(raw)
    np.testing.assert_array_equal(decoded, [[3.0, 4.0, 1.0, 0.016]])
    assert loads.paths == []
    with pytest.raises(
        RuntimeError, match="mojo event-recording decoder is unavailable"
    ) as refused:
        event_recordings.decode_nmnist_recording(raw, backend="mojo")
    assert isinstance(refused.value.__cause__, MojoIsaRefusal)
    event_recordings._load_native_library.cache_clear()


def _mojo_context_loads(path: Path) -> list[int]:
    """Return the lines of ``ctypes.CDLL`` calls in ``path`` that may load a Mojo library outside the guard.

    A call counts when the file lies under ``accel/mojo`` or its scope (the enclosing function, or for a module-level
    call the enclosing top-level statement) names Mojo: in a string such as a path part, an identifier, a parameter
    or the function's own name. A keyword such as ``mojo=False`` passed to another call does not count. A call is
    exempt when it is the ``else`` branch of a conditional whose other branch is :func:`load_mojo_library`.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    found: list[int] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "CDLL"
        ):
            continue
        parent = parents.get(node)
        if (
            isinstance(parent, ast.IfExp)
            and parent.orelse is node
            and isinstance(parent.body, ast.Call)
            and getattr(parent.body.func, "id", None) == "load_mojo_library"
        ):
            continue
        scope: ast.AST = node
        while scope in parents and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if parents[scope] is tree:
                break
            scope = parents[scope]
        if "/accel/mojo/" in path.as_posix() or "mojo" in " ".join(_names(scope)).lower():
            found.append(node.lineno)
    return found


def _names(scope: ast.AST) -> list[str]:
    """Return the strings, identifiers, parameters and function names in ``scope``, but no keyword-argument names."""
    names: list[str] = []
    for node in ast.walk(scope):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.append(node.value)
        elif isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.arg):
            names.append(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
    return names


def _package_modules() -> list[Path]:
    """Return the package's Python files that Git tracks, or every file outside a work tree.

    An ignored Mojo environment inside the package carries a Python library of another version,
    which the syntax-tree rule must not read.
    """
    repo = SRC.parents[1]
    tracked = tracked_files(repo, "src/sc_neurocore")
    if tracked is None:
        return sorted(SRC.rglob("*.py"))
    return sorted(repo / name for name in tracked if name.endswith(".py"))


def test_no_mojo_library_is_loaded_around_the_guard() -> None:
    """Outside the guard module, no ``ctypes.CDLL`` call remains where a Mojo library may be loaded."""
    violations = {
        str(path.relative_to(SRC)): lines
        for path in _package_modules()
        if path.name != "native_library.py" and (lines := _mojo_context_loads(path))
    }
    assert violations == {}
