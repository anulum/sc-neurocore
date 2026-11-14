# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The compiler routes stay in functions short enough to read whole

"""The compiler routes stay in functions of at most 50 lines, registered as before.

Measured before this was written: ``build_compiler_router`` was 91 lines, every
route of the module in one function. Splitting it must not change a route: the
same paths, in the same order, served by functions of the same names.
"""

from collections.abc import Iterable, Iterator
import ast
from pathlib import Path
from typing import Any

import sc_neurocore.studio.api.compiler as compiler_routes
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioRuntimeSettings

#: The owner's limit on one function's length, docstring included.
LIMIT = 50

#: The module's routes and their endpoint functions, in the order they are registered.
ROUTES = [
    ("/api/compile", "api_compile"),
    ("/api/models/compile", "api_model_compile"),
    ("/api/models/cosim", "api_model_cosim"),
    ("/api/nir/compile", "api_nir_compile"),
    ("/api/ir/build", "api_ir_build"),
    ("/api/ir/verify", "api_ir_verify"),
    ("/api/ir/replay", "api_ir_replay"),
    ("/api/ir/emit-sv", "api_ir_emit_sv"),
    ("/api/ir/emit-sv-direct", "api_ir_emit_sv_direct"),
]


def _span(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Return a function's length in lines, from its first decorator to its last line.

    Parameters
    ----------
    node : ast.FunctionDef or ast.AsyncFunctionDef
        The function.

    Returns
    -------
    int
        Its length, docstring included.
    """
    first = node.decorator_list[0].lineno if node.decorator_list else node.lineno
    return (node.end_lineno or node.lineno) - first + 1


def test_every_function_of_the_compiler_routes_fits_the_length_limit() -> None:
    """No function of the module, nested ones included, runs past the limit."""
    tree = ast.parse(Path(compiler_routes.__file__).read_text(encoding="utf-8"))
    lengths = {
        node.name: _span(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {name: length for name, length in lengths.items() if length > LIMIT} == {}


def _leaf_routes(routes: Iterable[Any]) -> Iterator[Any]:
    """Yield the leaf routes, descending into included routers.

    Parameters
    ----------
    routes : Iterable[Any]
        An application's or router's routes.

    Yields
    ------
    Any
        Each route that serves an endpoint.
    """
    for route in routes:
        original = getattr(route, "original_router", None)
        if original is not None and hasattr(original, "routes"):
            yield from _leaf_routes(original.routes)
        else:
            yield route


def test_the_compiler_routes_are_registered_as_before(tmp_path: Path) -> None:
    """The same paths, in the same order, served by functions of the same names.

    Parameters
    ----------
    tmp_path : Path
        Test-owned jobs and audit location.
    """
    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"), audit_log_path=str(tmp_path / "audit.jsonl")
    )
    app = create_app(settings)
    served = [
        (route.path, route.endpoint.__name__)
        for route in _leaf_routes(app.routes)
        if getattr(getattr(route, "endpoint", None), "__module__", "") == compiler_routes.__name__
    ]
    assert served == ROUTES
