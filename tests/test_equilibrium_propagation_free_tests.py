# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_equilibrium_propagation.py

"""Module-level tests from former test_equilibrium_propagation.py."""

from __future__ import annotations

from tests.equilibrium_propagation_support import *  # noqa: F403

def test_training_package_exports_equilibrium_propagation_surface() -> None:
    """The training package facade exposes the documented EP research surface."""
    assert "EPNetwork" in training.__all__
    assert training.EPNetwork is EPNetwork
def test_training_package_exports_ep_without_torch() -> None:
    """The NumPy EP surface remains selectable when Torch is unavailable."""
    finder = _BlockTorchFinder()
    original_torch = sys.modules.pop("torch", None)
    sys.meta_path.insert(0, finder)

    try:
        reloaded = importlib.reload(training)

        assert reloaded.HAS_TORCH is False
        assert reloaded.EPNetwork is EPNetwork
        assert "EPNetwork" in reloaded.__all__
    finally:
        sys.meta_path.remove(finder)
        if original_torch is not None:
            sys.modules["torch"] = original_torch
        importlib.reload(training)
