# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Engine import gate for binding tests

"""Engine import gate: skip locally without the engine, never mask CI.

Engine-binding tests call :func:`require_engine` at module scope before
importing ``sc_neurocore_engine``. In an environment without the
compiled engine (the pure-Python lane) the whole module skips cleanly
instead of dying at collection. Hosted CI builds the engine and exports
``SC_NEUROCORE_REQUIRE_ENGINE=1``, which turns a missing engine back
into a hard import error — a broken engine build can never hide behind
a green wall of skips.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

import pytest


def require_engine(module: str = "sc_neurocore_engine") -> ModuleType:
    """Import the engine module, skipping only where CI does not forbid it."""
    if os.environ.get("SC_NEUROCORE_REQUIRE_ENGINE") == "1":
        return importlib.import_module(module)
    imported: ModuleType = pytest.importorskip(module)
    return imported
