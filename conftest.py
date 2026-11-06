# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Root conftest — makes src/ and bridge/ importable and starts JuliaCall first.

JuliaCall starts before any test module can import Torch. It is left
uninitialised when starting it would resolve a Julia project that the run did
not ask to provision, so collecting tests never rewrites a shared project.
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parent
for subdir in ("src", "bridge"):
    p = _root / subdir
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _preload_juliacall_before_torch() -> str | None:
    """Initialise JuliaCall before Torch to avoid native runtime clashes.

    JuliaCall documents a crash risk when Torch is imported first. Pytest
    collects all modules in one interpreter, so a Torch-focused test collected
    before a Julia parity test can put the process into that unsafe order.
    A Julia project that the import would resolve again is left untouched
    unless the run requires Julia; the returned notice then says why.
    """
    if "juliacall" in sys.modules or importlib.util.find_spec("juliacall") is None:
        return None
    from tests.julia_requirement import disable_unrequested_julia_resolution

    notice = disable_unrequested_julia_resolution(os.environ)
    try:
        importlib.import_module("juliacall")
    except Exception:
        pass
    return notice


_JULIA_NOTICE = _preload_juliacall_before_torch()


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Report a Julia runtime left uninitialised; conftest output is otherwise captured."""
    if _JULIA_NOTICE is not None:
        terminalreporter.write_line(_JULIA_NOTICE)
