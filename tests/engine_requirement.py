# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Engine import gate for binding tests

"""Engine import gate: skip locally without the engine, never mask CI.

Engine-binding tests call :func:`require_engine` at module scope before
importing ``sc_neurocore_engine``. The gate targets the COMPILED
extension submodule ``sc_neurocore_engine.sc_neurocore_engine`` — the
pure-Python package wrapper imports even without the extension, so
gating the top-level name alone would let an engine-less environment
die at collection anyway.

Skip semantics are deliberately narrow: the module skips only when the
extension (or the engine package itself) is genuinely ABSENT. A present
extension that fails to load, or an absent third-party dependency
raised while resolving it, is a hard import error — never a skip.
Hosted CI builds the engine and exports
``SC_NEUROCORE_REQUIRE_ENGINE=1``, which turns even genuine absence
into a hard import error, so a broken engine build can never hide
behind a green wall of skips.
"""

from __future__ import annotations

import hashlib
import importlib
from importlib.machinery import PathFinder
import importlib.util
import os
from pathlib import Path
import sysconfig
from types import ModuleType

import pytest

ENGINE_EXTENSION_MODULE = "sc_neurocore_engine.sc_neurocore_engine"


def require_engine(module: str = ENGINE_EXTENSION_MODULE) -> ModuleType:
    """Import the compiled engine module or skip when it is truly absent.

    ``SC_NEUROCORE_REQUIRE_ENGINE=1`` disables the skip path entirely.
    Without it, the module's spec is resolved first: a missing engine
    package or missing extension file skips the calling test module,
    while any other failure — a present extension that cannot load, or
    a broken transitive dependency — propagates as a hard error.
    """
    if os.environ.get("SC_NEUROCORE_REQUIRE_ENGINE") == "1":
        return importlib.import_module(module)

    top_level = module.split(".")[0]
    try:
        spec = importlib.util.find_spec(module)
    except ModuleNotFoundError as exc:
        # find_spec imports parent packages to resolve a dotted name; a
        # ModuleNotFoundError naming the engine itself means genuine
        # absence, while one naming anything else is a broken
        # dependency and must stay a hard failure.
        if exc.name in (module, top_level):
            spec = None
        else:
            raise
    if spec is None:
        pytest.skip(
            f"compiled engine module {module!r} is not installed",
            allow_module_level=True,
        )
    return importlib.import_module(module)


def installed_engine_origins() -> dict[str, dict[str, str]]:
    """Resolve the engine files an isolated interpreter imports, with their digests.

    The repository's root ``conftest.py`` puts the checkout's ``bridge``
    directory first on ``sys.path``, so the test process itself may load the
    package wrapper from the source tree. A consumer started with ``-I`` sees
    only the interpreter's own installation directories, so the files it is
    expected to load are resolved there and never through ``sys.path``.

    Returns
    -------
    dict[str, dict[str, str]]
        The resolved ``path`` and ``sha256`` of the package wrapper and of the
        compiled extension, keyed by module name.

    Raises
    ------
    AssertionError
        The engine is not installed in the interpreter's installation
        directories, as with an editable or source-tree layout.
    """
    package = ENGINE_EXTENSION_MODULE.split(".")[0]
    roots = list(dict.fromkeys(sysconfig.get_path(name) for name in ("purelib", "platlib")))
    wrapper = PathFinder.find_spec(package, roots)
    assert wrapper is not None and wrapper.origin is not None, (
        f"{package!r} is not installed in the interpreter's installation directories {roots}"
    )
    extension = PathFinder.find_spec(ENGINE_EXTENSION_MODULE, [str(Path(wrapper.origin).parent)])
    assert extension is not None and extension.origin is not None, (
        f"{ENGINE_EXTENSION_MODULE!r} is not installed beside {wrapper.origin}"
    )
    origins = {}
    for name, origin in ((package, wrapper.origin), (ENGINE_EXTENSION_MODULE, extension.origin)):
        path = Path(origin).resolve()
        origins[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return origins
