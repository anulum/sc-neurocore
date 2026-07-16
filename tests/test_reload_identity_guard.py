# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — regression barrier for the autouse first-party reload guard

"""Lock the reload-without-restore fix in place.

Two layers are exercised:

* the reusable :func:`tests.reload_guard.restore_first_party_reloads` context manager, tested
  directly and hermetically (no cross-test ordering dependency); and
* the autouse fixture wired in :mod:`tests.conftest`, tested end to end by a polluting reload in
  one test whose fallout must not reach a sibling test.

If either the context manager or its autouse wiring regresses, one of these tests turns red, so
the latent nondeterministic red — a reloaded first-party class identity leaking across tests —
cannot silently return.
"""

from __future__ import annotations

import importlib
import string
import sys
from types import ModuleType

import sc_neurocore.world_model._lgssm_backends as lgssm_backends

from tests.module_reload import restore_module_namespace, snapshot_module_namespace
from tests.reload_guard import _is_first_party, restore_first_party_reloads

# Captured at collection time, before any test runs, so it is the pristine class object every
# by-value consumer of the module also holds.
_ORIGINAL_CFUNCTION = lgssm_backends._CFunction


def test_context_manager_restores_first_party_identity_after_reload() -> None:
    """A reload inside the guard leaves the module's original class objects in place on exit."""
    original = lgssm_backends._CFunction

    with restore_first_party_reloads():
        importlib.reload(lgssm_backends)
        # Inside the block the reload has genuinely rebound the class to a fresh object.
        assert lgssm_backends._CFunction is not original

    # On exit the original identity is restored, so by-value consumers stay valid.
    assert lgssm_backends._CFunction is original


def test_context_manager_snapshots_only_the_pre_first_reload_state() -> None:
    """Reloading the same module twice restores to the state before the *first* reload."""
    original = lgssm_backends._CFunction

    with restore_first_party_reloads():
        importlib.reload(lgssm_backends)
        first_reload_identity = lgssm_backends._CFunction
        importlib.reload(lgssm_backends)
        second_reload_identity = lgssm_backends._CFunction
        assert second_reload_identity is not first_reload_identity

    assert lgssm_backends._CFunction is original


def test_context_manager_passes_third_party_reload_through() -> None:
    """A non-first-party module is reloaded straight through and is *not* restored by the guard."""
    saved = snapshot_module_namespace(string)
    try:
        original_template = string.Template
        with restore_first_party_reloads():
            importlib.reload(string)
            reloaded_template = string.Template
            assert reloaded_template is not original_template
        # The guard restores only first-party modules, so ``string`` stays reloaded here.
        assert string.Template is reloaded_template
    finally:
        restore_module_namespace(string, saved)


def test_reloaded_then_removed_module_is_skipped_on_restore() -> None:
    """If a snapshotted module leaves ``sys.modules`` the restore loop skips it without error."""
    name = lgssm_backends.__name__
    saved_module = sys.modules[name]
    saved_namespace = snapshot_module_namespace(saved_module)
    try:
        with restore_first_party_reloads():
            importlib.reload(lgssm_backends)
            del sys.modules[name]
        # Reaching here without a KeyError proves the ``module is None`` branch was taken.
        assert sys.modules.get(name) is None
    finally:
        sys.modules[name] = saved_module
        restore_module_namespace(saved_module, saved_namespace)


def test_is_first_party_classifies_the_sc_neurocore_namespace() -> None:
    """The namespace test accepts the root and its subpackages and rejects everything else."""
    assert _is_first_party(lgssm_backends) is True
    assert _is_first_party(ModuleType("sc_neurocore")) is True
    assert _is_first_party(string) is False


def test_autouse_fixture_pollutes_without_manual_restore() -> None:
    """Polluter: reload a first-party module and deliberately do NOT restore it by hand.

    The autouse fixture wired in ``tests.conftest`` must clean this up before the next test.
    """
    importlib.reload(lgssm_backends)
    assert lgssm_backends._CFunction is not _ORIGINAL_CFUNCTION


def test_autouse_fixture_restored_identity_for_the_next_test() -> None:
    """Victim: the sibling polluter's un-restored reload must not have leaked to here."""
    assert lgssm_backends._CFunction is _ORIGINAL_CFUNCTION
