# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed GPU fixed-point batch admission and recovery

"""Require physical GPU execution and recoverable fixed-point output-shape refusals."""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
import pytest

from tests.engine_requirement import require_engine

_NATIVE = require_engine()


class _Runner(Protocol):
    def gpu_name(self) -> str:
        """Return the actual adapter selected for this batch runner."""
        ...

    def run(
        self,
        n_neurons: int,
        n_steps: int,
        leak_k: int,
        gain_k: int,
        currents: NDArray[np.int16],
        *,
        data_width: int = 16,
        fraction: int = 8,
        refractory_period: int = 2,
    ) -> tuple[NDArray[np.int32], NDArray[np.int16]]:
        """Advance independent fixed-point neurons with immutable input currents."""
        ...


@pytest.fixture
def runner() -> _Runner:
    """Require a real installed GPU runner without accepting a software fallback."""
    assert _NATIVE.GpuLifBatch.is_gpu_available()
    value = cast(_Runner, _NATIVE.GpuLifBatch())
    assert value.gpu_name() and not any(
        word in value.gpu_name().lower() for word in ("llvmpipe", "software", "basic render")
    )
    return value


@pytest.mark.parametrize(("n_neurons", "n_steps"), [(0, 17), (2, 0), (2, 32)])
def test_gpu_lif_outputs_match_public_cpu_reference(
    runner: _Runner, n_neurons: int, n_steps: int
) -> None:
    """Preserve exact spikes, voltages, empty shapes and independent returned arrays."""
    values = np.array([512, 1024][:n_neurons], dtype=np.int16)
    values.setflags(write=False)
    original = values.tobytes()
    expected = _NATIVE.batch_lif_run_multi(n_neurons, n_steps, 16, 256, values)
    actual = runner.run(n_neurons, n_steps, 16, 256, values)
    for index in range(2):
        result, reference = actual[index], expected[index]
        dtype = (np.int32, np.int16)[index]
        assert result.shape == (n_neurons, n_steps)
        assert result.dtype == dtype and result.flags.c_contiguous
        np.testing.assert_array_equal(result, reference)
        result[...] = -123
    repeated = runner.run(n_neurons, n_steps, 16, 256, values)
    for index in range(2):
        np.testing.assert_array_equal(repeated[index], expected[index])
    assert values.tobytes() == original and not values.flags.writeable


@pytest.mark.parametrize(
    ("n_neurons", "n_steps", "error", "message"),
    [
        (1, 2**32, ValueError, "GPU batch dimensions exceed u32 shader indexing"),
        (2, 2**31, ValueError, "GPU batch dimensions exceed u32 shader indexing"),
        (2, 2**64 - 1, MemoryError, "GPU batch output cannot be allocated"),
        (1, 2**30, MemoryError, "GPU batch output cannot be allocated"),
        (
            65535 * 64 + 1,
            1,
            ValueError,
            "GPU batch dispatch exceeds device workgroup limit",
        ),
    ],
)
def test_gpu_lif_output_refusal_recovers_on_same_runner(
    runner: _Runner, n_neurons: int, n_steps: int, error: type[Exception], message: str
) -> None:
    """Refuse oversized output before device allocation and retain healthy exact results."""
    healthy = np.array([512, 1024], dtype=np.int16)
    baseline = runner.run(2, 32, 16, 256, healthy)
    values = np.ones(n_neurons, dtype=np.int16)
    values.setflags(write=False)
    original = values.tobytes()
    with pytest.raises(error) as refusal:
        runner.run(n_neurons, n_steps, 16, 256, values)
    assert str(refusal.value) == message
    assert values.tobytes() == original and not values.flags.writeable
    for result, reference in zip(runner.run(2, 32, 16, 256, healthy), baseline, strict=True):
        np.testing.assert_array_equal(result, reference)


def test_gpu_lif_input_length_refusal_precedes_output_capacity(runner: _Runner) -> None:
    """Retain the input-length error before configuration and output admission."""
    with pytest.raises(ValueError) as refusal:
        runner.run(2, 2**64 - 1, 16, 256, np.ones(1, dtype=np.int16), data_width=0)
    assert str(refusal.value) == "currents length 1 does not match n_neurons 2"


@pytest.mark.parametrize(("n_neurons", "n_steps"), [(0, 8), (1, 0), (1, 8)])
@pytest.mark.parametrize(
    ("data_width", "fraction", "refractory_period", "message"),
    [
        (0, 8, 2, "data_width must be in [1, 16]"),
        (17, 8, 2, "data_width must be in [1, 16]"),
        (16, 16, 2, "fraction must be less than data_width"),
        (16, 8, -1, "refractory_period must be nonnegative"),
    ],
)
def test_gpu_lif_configuration_refusals_match_cpu_before_output(
    runner: _Runner,
    n_neurons: int,
    n_steps: int,
    data_width: int,
    fraction: int,
    refractory_period: int,
    message: str,
) -> None:
    """Require CPU configuration admission for empty and nonempty batches."""
    values = np.array([512][:n_neurons], dtype=np.int16)
    values.setflags(write=False)
    before = values.tobytes()
    baseline = runner.run(n_neurons, 8, 16, 256, values)
    with pytest.raises(ValueError) as cpu_error:
        _NATIVE.batch_lif_run_multi(
            n_neurons,
            n_steps,
            16,
            256,
            values,
            data_width=data_width,
            fraction=fraction,
            refractory_period=refractory_period,
        )
    with pytest.raises(ValueError) as gpu_error:
        runner.run(
            n_neurons,
            n_steps,
            16,
            256,
            values,
            data_width=data_width,
            fraction=fraction,
            refractory_period=refractory_period,
        )
    assert str(cpu_error.value) == str(gpu_error.value) == message
    assert values.tobytes() == before and not values.flags.writeable
    repeated = runner.run(n_neurons, 8, 16, 256, values)
    for index in range(2):
        np.testing.assert_array_equal(repeated[index], baseline[index])
