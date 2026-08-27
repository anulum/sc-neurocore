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
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from types import ModuleType

import pytest

JULIA_BRIDGE_MODULE = "juliacall"


def require_julia(module: str = JULIA_BRIDGE_MODULE) -> ModuleType:
    """Import the Julia bridge or skip only when it is truly absent.

    ``SC_NEUROCORE_REQUIRE_JULIA=1`` disables the skip path entirely.
    Without it, the module's spec is resolved first: a missing bridge
    package skips the calling test module, while any other failure —
    a present bridge that cannot load, or a broken transitive
    dependency — propagates as a hard error.
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
    return importlib.import_module(module)
