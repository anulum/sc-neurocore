# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed GPU Izhikevich batch admission and recovery

"""Require physical GPU execution and recoverable Izhikevich output-shape refusals."""

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
        self, n_neurons: int, n_steps: int, currents: NDArray[np.float32]
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """Advance independent regular-spiking neurons with immutable currents."""
        ...


@pytest.fixture
def runner() -> _Runner:
    """Require a real installed GPU runner without accepting a software fallback."""
    assert _NATIVE.GpuIzhikevichBatch.is_gpu_available()
    value = cast(_Runner, _NATIVE.GpuIzhikevichBatch())
    assert value.gpu_name() and not any(
        word in value.gpu_name().lower() for word in ("llvmpipe", "software", "basic render")
    )
    return value


@pytest.mark.parametrize(("n_neurons", "n_steps"), [(0, 17), (2, 0), (2, 8)])
def test_gpu_izhikevich_outputs_preserve_public_cpu_dynamics(
    runner: _Runner, n_neurons: int, n_steps: int
) -> None:
    """Compare public CPU spike events and f32 subthreshold voltage traces."""
    values = np.array([0.0, 0.5][:n_neurons], dtype=np.float32)
    values.setflags(write=False)
    original = values.tobytes()
    spikes, voltages = runner.run(n_neurons, n_steps, values)
    assert spikes.shape == voltages.shape == (n_neurons, n_steps)
    assert spikes.dtype == np.int32 and voltages.dtype == np.float32
    assert spikes.flags.c_contiguous and voltages.flags.c_contiguous
    for row, current in enumerate(values):
        reference = _NATIVE.Izhikevich()
        expected_spikes: list[int] = []
        expected_voltages: list[float] = []
        for _ in range(n_steps):
            expected_spikes.append(reference.step(float(current)))
            expected_voltages.append(reference.get_state()["v"])
        np.testing.assert_array_equal(spikes[row], expected_spikes)
        np.testing.assert_allclose(voltages[row], expected_voltages, rtol=0.0, atol=1e-3)
    baseline = (spikes.copy(), voltages.copy())
    spikes[...] = -123
    voltages[...] = 123.0
    for result, expected in zip(runner.run(n_neurons, n_steps, values), baseline, strict=True):
        np.testing.assert_array_equal(result, expected)
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
def test_gpu_izhikevich_output_refusal_recovers_on_same_runner(
    runner: _Runner, n_neurons: int, n_steps: int, error: type[Exception], message: str
) -> None:
    """Refuse oversized output before device allocation and retain healthy trajectories."""
    healthy = np.array([0.0, 0.5], dtype=np.float32)
    baseline = runner.run(2, 8, healthy)
    values = np.ones(n_neurons, dtype=np.float32)
    values.setflags(write=False)
    original = values.tobytes()
    with pytest.raises(error) as refusal:
        runner.run(n_neurons, n_steps, values)
    assert str(refusal.value) == message
    assert values.tobytes() == original and not values.flags.writeable
    for result, expected in zip(runner.run(2, 8, healthy), baseline, strict=True):
        np.testing.assert_array_equal(result, expected)


def test_gpu_izhikevich_input_length_refusal_precedes_output_capacity(runner: _Runner) -> None:
    """Retain the established input error before new output-capacity admission."""
    with pytest.raises(ValueError) as refusal:
        runner.run(2, 2**64 - 1, np.ones(1, dtype=np.float32))
    assert str(refusal.value) == "currents length 1 does not match n_neurons 2"
