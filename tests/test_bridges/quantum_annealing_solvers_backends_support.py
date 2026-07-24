# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former quantum_annealing_solvers_backends

from __future__ import annotations

"""Exercise optional dependency loading and solver result contracts."""

import builtins


import importlib


import sys


import types


from collections.abc import Mapping, Sequence


import pytest


from sc_neurocore.bridges import annealing_backends as backends


from sc_neurocore.bridges.quantum_annealing import (
    DWaveInterface,
    IsingModel,
    QUBOModel,
    SimulatedAnnealer,
)


from tests.test_bridges.quantum_annealing_test_helpers import simple_ising, unsafe


def _valid_native_result(size: int) -> dict[str, object]:
    """Return a complete two-sample native result."""
    first = [1 if index % 2 == 0 else -1 for index in range(size)]
    second = [-spin for spin in first]
    return {
        "best_spins": first,
        "best_energy": -3.0,
        "energies": [-3.0, -2.0],
        "samples": [first, second],
    }



__all__ = ['builtins', 'importlib', 'sys', 'types', 'Mapping', 'Sequence', 'pytest', 'backends', 'DWaveInterface', 'IsingModel', 'QUBOModel', 'SimulatedAnnealer', 'simple_ising', 'unsafe', '_valid_native_result']
