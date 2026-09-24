# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio platform lazy facade contract

"""The platform facade exports exactly its public names, lazily and by identity.

The start-up check runs a real, separate interpreter: loading every Studio
subsystem there is what pushed job workers and their lifetime guards past
their fixed readiness deadlines.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import sc_neurocore.studio.platform as platform

FACADE_SOURCE = Path(platform.__file__)
SOURCE_ROOT = FACADE_SOURCE.parents[3]


def _type_checking_imports() -> dict[str, str]:
    """Map each name imported for type checkers to its defining submodule."""
    tree = ast.parse(FACADE_SOURCE.read_text(encoding="utf-8"))
    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
    ]
    assert len(guards) == 1
    imports: dict[str, str] = {}
    for statement in guards[0].body:
        assert isinstance(statement, ast.ImportFrom) and statement.module is not None
        package, _, module = statement.module.rpartition(".")
        assert package == platform.__name__
        for alias in statement.names:
            assert alias.asname is None and alias.name not in imports
            imports[alias.name] = module
    return imports


def test_runtime_sources_match_type_checking_imports_and_all() -> None:
    """Type checkers, lazy resolution and ``__all__`` see the same names."""
    imports = _type_checking_imports()
    assert len(platform.__all__) == len(set(platform.__all__)) == 209
    assert set(platform.__all__) == set(imports)
    assert imports == platform._EXPORT_SOURCES


def test_every_public_name_is_the_defining_module_object() -> None:
    """Each resolved name is the submodule object and is cached on the facade."""
    for name in platform.__all__:
        source = importlib.import_module(f"{platform.__name__}.{platform._EXPORT_SOURCES[name]}")
        assert getattr(platform, name) is getattr(source, name), name
        assert vars(platform)[name] is getattr(source, name), name
    assert set(platform.__all__) <= set(dir(platform))


def test_unknown_names_raise_and_submodules_still_import() -> None:
    """Unknown attributes fail normally; submodule imports are unaffected."""
    with pytest.raises(AttributeError, match="has no attribute 'no_such_name'"):
        _ = platform.no_such_name
    from sc_neurocore.studio.platform import jobs_ledger_supervisor

    assert jobs_ledger_supervisor.__name__ == f"{platform.__name__}.jobs_ledger_supervisor"


def _loaded_after(statement: str) -> list[str]:
    """Return the ``sc_neurocore`` modules a fresh interpreter loads for ``statement``."""
    program = (
        f"{statement}\nimport json, sys\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('sc_neurocore'))))"
    )
    environment = {**os.environ, "PYTHONPATH": str(SOURCE_ROOT)}
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        check=True,
        env=environment,
        text=True,
        timeout=120.0,
    )
    loaded: list[str] = json.loads(completed.stdout)
    return loaded


def test_narrow_import_loads_only_its_own_dependencies() -> None:
    """The worker guard's liveness module loads no other Studio subsystem."""
    loaded = _loaded_after("import sc_neurocore.studio.platform.jobs_ledger_supervisor")
    assert loaded == [
        "sc_neurocore",
        "sc_neurocore.studio",
        "sc_neurocore.studio.platform",
        "sc_neurocore.studio.platform.jobs_ledger_supervisor",
    ]


def test_star_import_resolves_every_public_name() -> None:
    """``from ... import *`` still binds all names, loading their submodules."""
    loaded = _loaded_after("from sc_neurocore.studio.platform import *")
    sources = {f"{platform.__name__}.{module}" for module in platform._EXPORT_SOURCES.values()}
    assert sources <= set(loaded)
