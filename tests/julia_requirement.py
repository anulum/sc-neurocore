# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia bridge import gate for parity tests

"""Julia bridge gate: skip locally without juliacall, never mask CI.

Julia-lane parity tests call :func:`require_julia` at module scope.
The gate resolves the ``juliacall`` bridge the same way the engine
gate resolves the compiled extension: a genuinely ABSENT bridge skips
the whole module, while a present-but-broken bridge (or a broken
transitive dependency raised while resolving it) is a hard import
error — never a skip. Hosted CI installs and warms the bridge and
exports ``SC_NEUROCORE_REQUIRE_JULIA=1``, which turns even genuine
absence into a hard error, so a broken Julia toolchain can never hide
behind a green wall of skips. Kernel-level failures after a healthy
import (a missing ``.jl`` file, a Julia-side load error) stay hard
test failures in the test bodies.

The root ``conftest.py`` imports the bridge before any test. Outside a
run that requires Julia, :func:`disable_unrequested_julia_resolution`
first leaves it uninitialised when its Julia project would otherwise be
resolved and rewritten, so collecting tests from any checkout leaves a
shared virtual environment's project unchanged.
"""

from __future__ import annotations

from collections.abc import MutableMapping
import importlib
import importlib.util
import os
from types import ModuleType

import pytest

JULIA_BRIDGE_MODULE = "juliacall"


def disable_unrequested_julia_resolution(environ: MutableMapping[str, str]) -> str | None:
    """Keep JuliaCall from rewriting a Julia project that this run did not ask to provision.

    Importing ``juliacall`` makes ``juliapkg`` resolve its project again whenever
    the recorded metadata no longer matches the Julia executable or the
    ``juliapkg.json`` files on ``sys.path``. A checkout other than the one that
    last resolved a virtual environment's default project therefore rewrites the
    project that every other checkout of that environment shares. Call this
    before the first ``juliacall`` import.

    Parameters
    ----------
    environ : MutableMapping of str to str
        Environment that JuliaCall reads when it is first imported.

    Returns
    -------
    str or None
        The reason JuliaCall was left uninitialised, or None when the run
        requires Julia (``SC_NEUROCORE_REQUIRE_JULIA=1``), names its own Julia
        executable or directory together with a project, already disabled
        initialisation or has a current project.

    Notes
    -----
    A project that would be resolved again, or whose metadata cannot be read,
    sets ``PYTHON_JULIACALL_INIT=no`` in ``environ``, so the import writes
    nothing. The check reads the project metadata and runs ``julia --version``;
    it never takes the project lock.
    """
    own_runtime = "PYTHON_JULIACALL_EXE" in environ or "PYTHON_JULIACALL_BINDIR" in environ
    if (
        environ.get("PYTHON_JULIACALL_INIT") == "no"
        or environ.get("SC_NEUROCORE_REQUIRE_JULIA") == "1"
        or (own_runtime and "PYTHON_JULIACALL_PROJECT" in environ)
    ):
        return None
    deps = importlib.import_module("juliapkg.deps")
    try:
        current = bool(deps.can_skip_resolve())
    except Exception:
        # Damaged metadata cannot show the project is current; juliapkg would rewrite it.
        current = False
    if current:
        return None
    environ["PYTHON_JULIACALL_INIT"] = "no"
    return (
        f"JuliaCall stays uninitialised: importing it would resolve the Julia project "
        f"{deps.STATE['project']} again. Set SC_NEUROCORE_REQUIRE_JULIA=1 to let this run "
        "resolve it, after pointing PYTHON_JULIAPKG_PROJECT at a private copy when the "
        "project is shared."
    )


def require_julia(module: str = JULIA_BRIDGE_MODULE) -> ModuleType:
    """Import the Julia bridge or skip only when it is truly absent.

    ``SC_NEUROCORE_REQUIRE_JULIA=1`` disables the skip path entirely.
    Without it, the module's spec is resolved first: a missing bridge
    package skips the calling test module, while any other failure —
    a present bridge that cannot load, or a broken transitive
    dependency — propagates as a hard error. A bridge imported with
    initialisation disabled has no Julia runtime, so it also skips.
    """
    if os.environ.get("SC_NEUROCORE_REQUIRE_JULIA") == "1":
        return importlib.import_module(module)

    top_level = module.split(".")[0]
    try:
        spec = importlib.util.find_spec(module)
    except ModuleNotFoundError as exc:
        if exc.name in (module, top_level):
            spec = None
        else:
            raise
    if spec is None:
        pytest.skip(
            f"Julia bridge module {module!r} is not installed",
            allow_module_level=True,
        )
    bridge = importlib.import_module(module)
    if getattr(bridge, "CONFIG", {}).get("init") is False:
        pytest.skip(
            "JuliaCall was left uninitialised for this run (PYTHON_JULIACALL_INIT=no)",
            allow_module_level=True,
        )
    return bridge
