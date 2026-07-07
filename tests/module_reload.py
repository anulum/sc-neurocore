# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — hermetic module-reload helpers for tests

"""Hermetic ``importlib.reload`` support for tests that probe optional-backend branches.

One responsibility: let a test reload a module to exercise a fallback branch without
leaking *new* class identities to the rest of the suite.

``importlib.reload`` re-executes the target in its existing ``__dict__``, rebinding every
class to a fresh object. "Restoring" with a second ``importlib.reload`` leaves those new
identities in place — so any module that imported the original classes *by value* at
collection time (e.g. ``from mod import Thing``) then fails ``isinstance(x, Thing)`` against
objects the reloaded code builds, because the producer resolves ``Thing`` from the mutated
module namespace. Under CI's ``pytest -n auto --dist loadfile`` the polluter and victim land
on different xdist workers and never collide; a serial ``pytest tests/`` (or any co-located
run) fails deterministically. Snapshotting the namespace and putting the *original* objects
back on teardown keeps the canonical identities other modules hold valid.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from types import ModuleType


def snapshot_module_namespace(module: ModuleType) -> dict[str, object]:
    """Shallow-copy a module namespace so its original class objects can be restored later."""
    return dict(vars(module))


def restore_module_namespace(module: ModuleType, snapshot: dict[str, object]) -> None:
    """Restore a module namespace to ``snapshot`` in place (the module object is preserved)."""
    vars(module).clear()
    vars(module).update(snapshot)


@contextlib.contextmanager
def preserve_module_identity(module: ModuleType) -> Iterator[None]:
    """Restore ``module``'s original class identities after a reload-based test.

    Wrap the ``importlib.reload`` + assertions so that, however the module is mutated inside,
    the exact class objects other modules imported by value are put back on exit.
    """
    snapshot = snapshot_module_namespace(module)
    try:
        yield
    finally:
        restore_module_namespace(module, snapshot)
