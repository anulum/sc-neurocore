# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_equilibrium_propagation.py

from __future__ import annotations

"""Test suite for the EP research prototype."""
import importlib
import importlib.abc
import sys
import numpy as np
import sc_neurocore.training as training
from sc_neurocore.training.equilibrium_propagation import EPNetwork, _rho, _rho_prime


class _BlockTorchFinder(importlib.abc.MetaPathFinder):
    """Import hook that forces the training package through its no-Torch branch."""

    def find_spec(
        self,
        fullname: str,
        path: object | None,
        target: object | None = None,
    ) -> None:
        if fullname == "torch":
            raise ImportError("forced missing torch surface")
        return None


__all__ = [
    "importlib",
    "sys",
    "np",
    "training",
    "EPNetwork",
    "_rho",
    "_rho_prime",
    "_BlockTorchFinder",
]
