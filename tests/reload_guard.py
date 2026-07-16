# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — autouse guard that restores first-party module identity after a test reload

"""Contain ``importlib.reload`` fallout so no reloaded first-party class identity leaks.

One responsibility: for the duration of a block (a single test), snapshot every first-party
``sc_neurocore`` module the moment it is first reloaded and put its original namespace back on
exit — turning the whole *reload-without-restore* class into a no-op regardless of whether the
individual test remembered to restore.

``importlib.reload`` re-executes a module in its existing ``__dict__``, rebinding every class,
function and constant to a fresh object. Any consumer that imported a symbol *by value*
(``from mod import Thing``) keeps the original object, so after an un-restored reload
``mod.Thing is not Thing``; an ``isinstance`` / identity check against the by-value import then
fails for objects the reloaded code builds. Under CI's ``pytest -n auto --dist loadfile`` the
polluting reload and its victim land on different workers and never collide, but a serial run
(or any co-located ordering) fails deterministically — a latent nondeterministic red. Note that
pytest's ``--import-mode=importlib`` does **not** help: it changes how *test* modules are
imported, not how a *production* module is reloaded, so the class-rebind is identical under it.

Restoring by a second ``importlib.reload`` does not work either — it installs yet another fresh
identity. The only fix is to snapshot the namespace before the first reload and restore the
*original* objects afterwards; that is what this guard does, centrally, for every first-party
module, so the pollution cannot escape a test even if the test body forgets to clean up.

The complementary :mod:`tests.module_reload` helpers remain available for a test that wants to
scope the restore explicitly to one module inside its own body; this guard is the belt-and-
braces safety net wired in :mod:`tests.conftest` as an autouse fixture.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from collections.abc import Iterator
from types import ModuleType

_FIRST_PARTY_ROOT = "sc_neurocore"


def _is_first_party(module: ModuleType) -> bool:
    """Return whether ``module`` belongs to the first-party ``sc_neurocore`` namespace."""
    name = getattr(module, "__name__", "")
    return name == _FIRST_PARTY_ROOT or name.startswith(f"{_FIRST_PARTY_ROOT}.")


@contextlib.contextmanager
def restore_first_party_reloads() -> Iterator[None]:
    """Restore the original identities of any first-party module reloaded inside the block.

    While the block is active, ``importlib.reload`` is wrapped so that the first time each
    first-party module is reloaded its namespace is snapshotted; on exit every snapshotted
    module has its original objects put back in place (``vars(module)`` cleared and repopulated,
    so the module object itself — and every by-value reference other modules hold to it — stays
    valid). Reloads of third-party modules are passed straight through untouched.

    Yields
    ------
    None
        Control returns to the caller with the guard active for the duration of the block.
    """
    original_reload = importlib.reload
    snapshots: dict[str, dict[str, object]] = {}

    def guarded_reload(module: ModuleType) -> ModuleType:
        if _is_first_party(module):
            name = module.__name__
            if name not in snapshots:
                snapshots[name] = dict(vars(module))
        return original_reload(module)

    importlib.reload = guarded_reload
    try:
        yield
    finally:
        importlib.reload = original_reload
        for name, snapshot in snapshots.items():
            module = sys.modules.get(name)
            if module is None:
                continue
            module_namespace = vars(module)
            module_namespace.clear()
            module_namespace.update(snapshot)
