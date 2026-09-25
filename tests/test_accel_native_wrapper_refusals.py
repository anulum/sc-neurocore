# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native kernel wrappers refuse what their kernels refuse

"""The Go and Mojo wrappers turn a kernel's refusal into an error, never a trace.

Each native kernel returns a negative count when its simulation contract is
broken — here by a non-finite drive — and the Python wrapper must raise rather
than hand back the untouched output buffer as if it were a run. The complete
Quadratic IF wrappers also refuse to run on a backend that was never loaded.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from sc_neurocore.accel import lapicque, quadratic_if, theta

NAN = math.nan


def _loaded(loader: Callable[[], bool], name: str) -> None:
    if not loader():
        pytest.skip(f"{name} backend is not built in this environment")


LAPICQUE = (0.0, 0.0, 0.0, 1.0, 20.0, 1.0, 1.0)
LAPICQUE_SOURCE = (1.1, 1.0, 1.0, False, False)


@pytest.mark.parametrize(
    ("language", "call"),
    [
        ("go", lambda: lapicque.simulate_go(*LAPICQUE, 3, NAN)),
        ("mojo", lambda: lapicque.simulate_mojo(*LAPICQUE, 3, NAN)),
        ("go", lambda: lapicque.simulate_go_complete(*LAPICQUE, *LAPICQUE_SOURCE, 3, NAN)),
        ("mojo", lambda: lapicque.simulate_mojo_complete(*LAPICQUE, *LAPICQUE_SOURCE, 3, NAN)),
    ],
    ids=["go", "mojo", "go-complete", "mojo-complete"],
)
def test_lapicque_wrappers_raise_on_a_kernel_refusal(
    language: str, call: Callable[[], object]
) -> None:
    loader = lapicque.ensure_go_loaded if language == "go" else lapicque.ensure_mojo_loaded
    _loaded(loader, f"{language} Lapicque")
    with pytest.raises(
        FloatingPointError, match="Lapicque kernel rejected the simulation contract"
    ):
        call()


@pytest.mark.parametrize(
    ("module", "language", "label"),
    [
        (theta, "go", "Theta"),
        (theta, "mojo", "Theta"),
        (quadratic_if, "go", "Quadratic IF"),
        (quadratic_if, "mojo", "Quadratic IF"),
    ],
)
def test_single_trace_wrappers_raise_on_a_kernel_refusal(
    module: Any, language: str, label: str
) -> None:
    loader = module.ensure_go_loaded if language == "go" else module.ensure_mojo_loaded
    _loaded(loader, f"{language} {label}")
    simulate = module.simulate_go if language == "go" else module.simulate_mojo
    arguments = (0.0, 0.1) if module is theta else (0.0, -1.0, 1.0, 0.1)
    with pytest.raises(
        FloatingPointError, match=f"{label} kernel rejected the simulation contract"
    ):
        simulate(*arguments, 3, NAN)


def _fresh_quadratic_if() -> ModuleType:
    """A second copy of the backend module, in which no backend has been loaded yet."""
    path = Path(quadratic_if.__file__)
    spec = importlib.util.spec_from_file_location("sc_neurocore_fresh_quadratic_if", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("function", "message"),
    [
        ("simulate_julia_complete", "Julia Quadratic IF module is unavailable"),
        ("simulate_go_complete", "Go Quadratic IF library is unavailable"),
        ("simulate_mojo_complete", "Mojo Quadratic IF library is unavailable"),
    ],
)
def test_complete_wrappers_refuse_a_backend_that_was_never_loaded(
    function: str, message: str
) -> None:
    fresh = _fresh_quadratic_if()
    with pytest.raises(RuntimeError, match=message):
        getattr(fresh, function)(0.0, -1.0, 1.0, 0.1, False, 3, 1.0)


def test_the_rust_complete_wrapper_refuses_without_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``None`` module entry is how the import system records an absent package.

    The engine is installed wherever this suite runs, so its absence can only be
    produced by marking it absent before the module is loaded.
    """
    monkeypatch.setitem(sys.modules, "sc_neurocore_engine", None)
    fresh = _fresh_quadratic_if()
    assert fresh._HAS_RUST is False
    with pytest.raises(RuntimeError, match="Rust Quadratic IF complete batch is unavailable"):
        fresh.simulate_rust_complete(0.0, -1.0, 1.0, 0.1, False, 3, 1.0)
