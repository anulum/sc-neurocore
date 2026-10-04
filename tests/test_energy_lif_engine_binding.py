# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — installed EnergyLIF PyO3 contracts

"""Exercise the configured EnergyLIF class and native array batch boundary."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
import pytest

from sc_neurocore.neurons.models.energy_lif import EnergyLIFNeuron
from tests.energy_lif_contract_support import (
    CONFIGURATIONS,
    INVALID_CONFIGURATIONS,
    PARAMETERS,
    configuration_values,
)
from tests.engine_requirement import require_engine

extension = require_engine()
import sc_neurocore_engine


class NativeNeuron(Protocol):
    """Describe the actual exposed native cell methods."""

    def step(self, current: object) -> int:
        """Advance a scalar current or refuse without changing state."""
        ...

    def reset(self) -> None:
        """Restore the configured source equilibrium."""
        ...

    def get_state(self) -> dict[str, float]:
        """Return a fresh dictionary containing both dynamic states."""
        ...


class Constructor(Protocol):
    """Describe positional and keyword construction of the native class."""

    def __call__(self, *args: object, **kwargs: object) -> NativeNeuron:
        """Construct a cell from the sixteen exposed parameters."""
        ...


_CONSTRUCTOR = cast(Constructor, sc_neurocore_engine.EnergyLIFNeuron)
_BATCH = cast(Callable[..., dict[str, object]], extension.py_energy_lif_simulate)


@pytest.mark.parametrize("parameters", CONFIGURATIONS)
def test_configured_class_batch_trace_and_reset(parameters: dict[str, float]) -> None:
    """Compare every configured field with the real Python model and native batch."""
    reference = EnergyLIFNeuron(**parameters)
    native = _CONSTRUCTOR(**parameters)
    positional = _CONSTRUCTOR(*configuration_values(parameters))
    drive = np.resize(np.array([80.0, 0.0, 120.0, 20.0]), 256)
    batch = _BATCH(*configuration_values(parameters), drive)
    voltages: list[float] = []
    energies: list[float] = []
    events: list[int] = []
    for current in drive:
        event = reference.step(float(current))
        assert native.step(current) == positional.step(current) == event
        assert type(event) is int
        assert native.get_state() == positional.get_state()
        assert native.get_state() == pytest.approx(
            {"v": reference.v, "epsilon": reference.epsilon}, rel=0, abs=2e-12
        )
        voltages.append(reference.v)
        energies.append(reference.epsilon)
        events.append(event)
    np.testing.assert_array_equal(batch["events"], events)
    for key, expected in (("voltages", voltages), ("epsilon", energies)):
        actual = np.asarray(batch[key])
        assert actual.dtype == np.float64 and actual.shape == drive.shape
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
    assert np.asarray(batch["events"]).dtype == np.int32
    assert batch["v_final"] == native.get_state()["v"]
    assert batch["epsilon_final"] == native.get_state()["epsilon"]
    reference.reset()
    native.reset()
    assert native.get_state() == {"v": reference.v, "epsilon": reference.epsilon}
    for current in drive[:32]:
        assert native.step(current) == reference.step(float(current))
        assert native.get_state() == pytest.approx(
            {"v": reference.v, "epsilon": reference.epsilon}, rel=0, abs=2e-12
        )


@pytest.mark.parametrize("parameters", INVALID_CONFIGURATIONS)
@pytest.mark.parametrize("count", [0, 1])
def test_invalid_configuration_is_refused_before_any_batch(
    parameters: dict[str, float], count: int
) -> None:
    """Refuse unsafe reset domains and integration parameters for empty batches too."""
    with pytest.raises(ValueError, match="^invalid EnergyLIF state or configuration$"):
        _CONSTRUCTOR(**parameters)
    drive = np.full(count, 80.0)
    before = drive.copy()
    with pytest.raises(ValueError, match="^invalid EnergyLIF state or configuration$"):
        _BATCH(*configuration_values(parameters), drive)
    np.testing.assert_array_equal(drive, before)
    assert _CONSTRUCTOR().step(80.0) == 0


@pytest.mark.parametrize("field", [name for name, _, _ in PARAMETERS])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_every_nonfinite_configuration_field_is_refused(field: str, value: float) -> None:
    """Validate each actual scalar extraction before constructor or zero-work batch."""
    parameters = {field: value}
    with pytest.raises(ValueError, match="^invalid EnergyLIF state or configuration$"):
        _CONSTRUCTOR(**parameters)
    with pytest.raises(ValueError, match="^invalid EnergyLIF state or configuration$"):
        _BATCH(*configuration_values(parameters), np.empty(0))


@pytest.mark.parametrize("current", [float("nan"), float("inf"), -float("inf"), 1e308])
def test_refused_transition_preserves_native_state_and_retry(current: float) -> None:
    """Compare a real post-refusal continuation with an untouched control cell."""
    native = _CONSTRUCTOR()
    control = _CONSTRUCTOR()
    before = native.get_state()
    with pytest.raises(ValueError):
        native.step(current)
    assert native.get_state() == before
    assert native.step(80.0) == control.step(80.0)
    assert native.get_state() == control.get_state()


def test_post_spike_energy_refusal_preserves_native_state() -> None:
    """An unaffordable event leaves both native dynamic states uncommitted."""
    native = _CONSTRUCTOR(v=-58.8, delta=1.0)
    before = native.get_state()
    with pytest.raises(ValueError, match="^EnergyLIF post-spike energy outside safety envelope$"):
        native.step(0.0)
    assert native.get_state() == before
    native.reset()
    assert native.get_state() == {"v": -62.5, "epsilon": 0.5}


def test_returned_state_and_batch_outputs_are_independent() -> None:
    """Caller writes to returned values cannot alter the native cell or input drive."""
    native = _CONSTRUCTOR()
    state = native.get_state()
    state["v"] = 99.0
    assert native.get_state()["v"] == -61.0
    drive = np.array([80.0, 0.0])
    result = _BATCH(*configuration_values({}), drive)
    voltages = np.asarray(result["voltages"])
    assert not np.shares_memory(voltages, drive)
    voltages[:] = 99.0
    np.testing.assert_array_equal(drive, [80.0, 0.0])


@pytest.mark.parametrize("drive", [np.ones(2, dtype=np.float32), np.ones((2, 1)), [80.0]])
def test_native_batch_requires_rank_one_float64_array(drive: object) -> None:
    """Refuse dtype, rank and Python-sequence inputs at the actual NumPy boundary."""
    with pytest.raises(TypeError):
        _BATCH(*configuration_values({}), drive)


def test_native_batch_layout_refusal_and_readonly_input() -> None:
    """Refuse strided arrays while accepting an unchanged contiguous readonly view."""
    drive = np.array([80.0, 0.0, 120.0, 20.0])
    with pytest.raises(TypeError, match="not contiguous or is misaligned"):
        _BATCH(*configuration_values({}), drive[::2])
    drive.setflags(write=False)
    result = _BATCH(*configuration_values({}), drive)
    assert np.asarray(result["events"]).shape == (4,)
    np.testing.assert_array_equal(drive, [80.0, 0.0, 120.0, 20.0])


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_global_identity_and_instance_refusal(protocol: int) -> None:
    """Round-trip the public class global and retain deliberate instance refusal."""
    assert pickle.loads(pickle.dumps(sc_neurocore_engine.EnergyLIFNeuron, protocol)) is (
        sc_neurocore_engine.EnergyLIFNeuron
    )
    native = _CONSTRUCTOR()
    before = native.get_state()
    with pytest.raises(TypeError):
        pickle.dumps(native, protocol)
    assert native.get_state() == before
    assert native.step(80.0) == 0


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])
def test_instance_copy_refusal_retains_state(copier: Callable[[object], object]) -> None:
    """Exercise the actual copy protocol and a subsequent valid native step."""
    native = _CONSTRUCTOR()
    before = native.get_state()
    with pytest.raises(TypeError):
        copier(native)
    assert native.get_state() == before
    assert native.step(80.0) == 0
