# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed retained normalized energy-LIF serialization and copy refusal contracts

"""Pin real global identities and instance serialization refusal without mutation."""

from __future__ import annotations

from collections.abc import Callable
import copy
import pickle
from typing import Protocol, cast

import pytest

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine


class NativeCell(Protocol):
    """Describe the actual public state used by serialization consumers."""

    def step(self, current: float) -> int:
        """Advance a finite current before testing persistence refusal."""
        ...

    def get_state(self) -> dict[str, float]:
        """Return all dynamic state values after the refused operation."""
        ...


_CONSTRUCTOR = cast(Callable[..., NativeCell], sc_neurocore_engine.SCNormalizedEnergyLIFNeuron)


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_global_identity_and_instance_serialization_refusal(protocol: int) -> None:
    """Resolve the original global while refusing instance pickle without mutation."""
    cls = sc_neurocore_engine.SCNormalizedEnergyLIFNeuron
    assert cls.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert pickle.loads(pickle.dumps(cls, protocol=protocol)) is cls
    cell = _CONSTRUCTOR()
    cell.step(0.7)
    before = cell.get_state()
    name = (
        "SCNormalizedEnergyLIFNeuron"
        if protocol < 2
        else f"{cls.__module__}.SCNormalizedEnergyLIFNeuron"
    )
    with pytest.raises(TypeError) as error:
        pickle.dumps(cell, protocol=protocol)
    assert str(error.value) == f"cannot pickle '{name}' object"
    assert cell.get_state() == before


@pytest.mark.parametrize("operation", [copy.copy, copy.deepcopy])
def test_instance_copy_refusal_preserves_state(operation: Callable[[NativeCell], object]) -> None:
    """Retain exact copy refusal and the configured temporal state."""
    cell = _CONSTRUCTOR(v=-65.0, epsilon=0.5)
    before = cell.get_state()
    with pytest.raises(TypeError) as error:
        operation(cell)
    assert (
        str(error.value)
        == "cannot pickle 'sc_neurocore_engine.sc_neurocore_engine.SCNormalizedEnergyLIFNeuron' object"
    )
    assert cell.get_state() == before
