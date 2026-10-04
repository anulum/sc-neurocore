# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed fixed-point LIF input contracts

"""Exercise configuration refusals, input precedence and state atomicity."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine

INVALID_CONFIGURATIONS = [
    ("data_width", 0, "data_width must be in [1, 16]"),
    ("data_width", 17, "data_width must be in [1, 16]"),
    ("data_width", 32, "data_width must be in [1, 16]"),
    ("data_width", 33, "data_width must be in [1, 16]"),
    ("data_width", 2**32 - 1, "data_width must be in [1, 16]"),
    ("fraction", 16, "fraction must be less than data_width"),
    ("fraction", 32, "fraction must be less than data_width"),
    ("fraction", 2**32 - 1, "fraction must be less than data_width"),
    ("refractory_period", -1, "refractory_period must be nonnegative"),
    ("refractory_period", -(2**31), "refractory_period must be nonnegative"),
]


@pytest.mark.parametrize(("field", "value", "message"), INVALID_CONFIGURATIONS)
def test_constructor_refuses_invalid_configuration(field: str, value: int, message: str) -> None:
    """Refuse unusable state before a caller can enter an aborting step."""
    with pytest.raises(ValueError) as refusal:
        engine.FixedPointLif(**{field: value})
    assert str(refusal.value) == message


def _call_batch(
    case: str,
    parameters: dict[str, int],
    currents: NDArray[np.int16],
    noises: NDArray[np.int16],
) -> None:
    """Reach the named public entry point with its actual input arrays."""
    if case.startswith("constant"):
        engine.batch_lif_run(currents.size, 0, 256, 3, **parameters)
    elif case.startswith("parallel"):
        steps = 0 if case == "parallel_zero_steps" else 2
        engine.batch_lif_run_multi(currents.size, steps, 0, 256, currents, **parameters)
    else:
        engine.batch_lif_run_varying(0, 256, currents, noises, **parameters)


@pytest.mark.parametrize(("field", "value", "message"), INVALID_CONFIGURATIONS)
@pytest.mark.parametrize(
    "case",
    [
        "constant_empty",
        "constant_nonempty",
        "parallel_zero_rows",
        "parallel_zero_steps",
        "parallel_nonempty",
        "varying_empty",
        "varying_nonempty",
    ],
)
def test_batch_configuration_refusal_covers_empty_and_nonempty_paths(
    field: str, value: int, message: str, case: str
) -> None:
    """Empty results cannot admit invalid configurations or mutate inputs."""
    values = [] if case.endswith("empty") and not case.endswith("nonempty") else [-2, 3]
    if case == "parallel_zero_rows":
        values = []
    currents = np.asarray(values, dtype=np.int16)
    noises = np.zeros(currents.size, dtype=np.int16)
    currents.setflags(write=False)
    noises.setflags(write=False)
    with pytest.raises(ValueError) as refusal:
        _call_batch(case, {field: value}, currents, noises)
    assert str(refusal.value) == message
    np.testing.assert_array_equal(currents, values)
    np.testing.assert_array_equal(noises, np.zeros(currents.size, dtype=np.int16))
    assert not currents.flags.writeable
    assert not noises.flags.writeable


@pytest.mark.parametrize(
    ("case", "message"),
    [
        (
            "parallel_stride",
            "Cannot read currents: The given array is not contiguous or is misaligned.",
        ),
        (
            "varying_stride",
            "Cannot read currents: The given array is not contiguous or is misaligned.",
        ),
        (
            "varying_noise_stride",
            "Cannot read noises: The given array is not contiguous or is misaligned.",
        ),
        ("parallel_length", "currents length 1 does not match n_neurons 2."),
        ("varying_length", "noises length 1 does not match currents length 2."),
    ],
)
def test_input_layout_and_length_refusals_precede_configuration(case: str, message: str) -> None:
    """Retain existing layout and length errors even with invalid width."""
    current_storage = np.array([1, 2, 3, 4], dtype=np.int16)
    noise_storage = np.array([5, 6, 7, 8], dtype=np.int16)
    currents = current_storage[::2] if case.endswith("stride") else current_storage[:2]
    noises = noise_storage[:2]
    if case == "varying_noise_stride":
        currents = current_storage[:2]
        noises = noise_storage[::2]
    if case == "parallel_length":
        currents = current_storage[:1]
    if case == "varying_length":
        noises = noise_storage[:1]
    current_storage.setflags(write=False)
    noise_storage.setflags(write=False)
    currents.setflags(write=False)
    noises.setflags(write=False)
    with pytest.raises(ValueError) as refusal:
        if case.startswith("parallel"):
            engine.batch_lif_run_multi(2, 0, 0, 0, currents, data_width=0)
        else:
            engine.batch_lif_run_varying(0, 0, currents, noises, data_width=0)
    assert str(refusal.value) == message
    np.testing.assert_array_equal(current_storage, [1, 2, 3, 4])
    np.testing.assert_array_equal(noise_storage, [5, 6, 7, 8])
    assert not currents.flags.writeable
    assert not noises.flags.writeable


@pytest.mark.parametrize("argument", ["leak_k", "gain_k", "i_t", "noise_in"])
@pytest.mark.parametrize(("value", "exception"), [(32768, OverflowError), ("invalid", TypeError)])
def test_step_argument_refusal_preserves_existing_neuron_state(
    argument: str, value: object, exception: type[Exception]
) -> None:
    """Failed integer extraction cannot consume a refractory step."""
    neuron = engine.FixedPointLif(refractory_period=3)
    assert neuron.step(0, 256, 300) == (1, 0)
    before = neuron.get_state()
    parameters: dict[str, object] = {"leak_k": 0, "gain_k": 256, "i_t": 3, "noise_in": 0}
    parameters[argument] = value
    with pytest.raises(exception):
        neuron.step(**parameters)
    assert neuron.get_state() == before
    assert neuron.step(0, 256, 3) == (0, 0)
    assert neuron.get_state() == {"v": 0, "refractory_counter": 2}


def test_configuration_refusal_precedes_impossible_output_allocation() -> None:
    """Invalid width is reported before dimensions reach the NumPy allocator."""
    with pytest.raises(ValueError) as refusal:
        engine.batch_lif_run(int(np.iinfo(np.intp).max) + 1, 0, 0, 0, data_width=0)
    assert str(refusal.value) == "data_width must be in [1, 16]"


@pytest.mark.parametrize("steps", [0, 2])
@pytest.mark.parametrize("interface", ["constant", "parallel", "varying"])
def test_valid_batch_outputs_are_typed_owned_arrays(steps: int, interface: str) -> None:
    """New fallible allocation preserves actual values and fresh array storage."""
    currents = np.full(steps, 3, dtype=np.int16)
    currents.setflags(write=False)
    if interface == "constant":
        spikes, voltages = engine.batch_lif_run(steps, 0, 256, 3)
    elif interface == "parallel":
        spikes, voltages = engine.batch_lif_run_multi(steps, 2, 0, 256, currents)
    else:
        spikes, voltages = engine.batch_lif_run_varying(0, 256, currents)
    shape = (steps, 2) if interface == "parallel" else (steps,)
    assert spikes.shape == voltages.shape == shape
    assert spikes.dtype == np.int32
    assert voltages.dtype == np.int16
    np.testing.assert_array_equal(spikes, np.zeros(shape, dtype=np.int32))
    expected = np.tile([3, 6], (steps, 1)) if interface == "parallel" else [3, 6][:steps]
    np.testing.assert_array_equal(voltages, expected)
    for output in (spikes, voltages):
        assert output.flags.c_contiguous and output.flags.aligned
        assert output.flags.owndata and output.flags.writeable
        assert not np.shares_memory(output, currents)
    assert not np.shares_memory(spikes, voltages)
    assert not currents.flags.writeable
    np.testing.assert_array_equal(currents, np.full(steps, 3, dtype=np.int16))
