# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Caller random-generator custody during conversion

"""Conversion must leave the caller's global random generators exactly as it found them.

User ``forward`` code runs while conversion traces and calibrates. The models
below draw from Python, NumPy and PyTorch generators inside that code, and two
threads convert at once with a pause inside ``forward`` so their sections would
interleave without serialisation.
"""

import random
import threading
import time

import numpy as np
import pytest
import torch
from torch import nn

from sc_neurocore.conversion import convert
from sc_neurocore.conversion.calibration import calibrate_activation_thresholds
from sc_neurocore.conversion.random_custody import preserved_random_state


class DrawingNetwork(nn.Module):
    """A dense ReLU network whose forward draws from every global generator.

    Parameters
    ----------
    pause : float
        Seconds to wait after drawing, releasing the interpreter to other threads.
    fail : bool
        Raise after drawing, as a broken user forward would.
    """

    def __init__(self, pause: float = 0.0, fail: bool = False) -> None:
        """Build two seeded dense layers."""
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        self.first = nn.Linear(3, 4)
        self.second = nn.Linear(4, 2)
        with torch.no_grad():
            for layer in (self.first, self.second):
                layer.weight.copy_(torch.rand(layer.weight.shape, generator=generator))
                layer.bias.zero_()
        self.pause = pause
        self.fail = fail

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw from Python, NumPy and PyTorch, then run the dense path."""
        random.random()
        np.random.random_sample()
        torch.rand(1)
        time.sleep(self.pause)
        if self.fail:
            raise TypeError("user forward failed after drawing")
        result: torch.Tensor = self.second(torch.relu(self.first(x)))
        return result


def snapshot() -> tuple[object, object, torch.Tensor]:
    """Return the current Python, NumPy and PyTorch CPU generator states."""
    torch_state: torch.Tensor = torch.random.get_rng_state()
    return random.getstate(), np.random.get_state(), torch_state


def assert_unchanged(before: tuple[object, object, torch.Tensor]) -> None:
    """Require every global generator to hold exactly its earlier state."""
    python_state, numpy_state, torch_state = snapshot()
    assert python_state == before[0]
    numpy_before = before[1]
    assert isinstance(numpy_state, tuple) and isinstance(numpy_before, tuple)
    assert numpy_state[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_state[1], numpy_before[1])
    assert numpy_state[2:] == numpy_before[2:]
    assert torch.equal(torch_state, before[2])


def test_conversion_restores_every_generator_the_forward_draws_from() -> None:
    """Tracing and calibration run user draws; the caller's states are unchanged."""
    model = DrawingNetwork()
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    before = snapshot()
    converted = convert(
        model,
        calibration_data=torch.rand(8, 3, generator=torch.Generator().manual_seed(4)),
    )
    assert_unchanged(before)
    assert converted.run([0.2, 0.4, 0.6], input_mode="constant").shape == (2,)


def test_concurrent_conversions_do_not_interleave_their_restores() -> None:
    """Two threads converting at once leave the generators as they were before both."""
    models = [[DrawingNetwork(pause=0.02) for _ in range(3)] for _ in range(2)]
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    before = snapshot()
    barrier = threading.Barrier(2)
    failures: list[BaseException] = []
    data = torch.rand(8, 3, generator=torch.Generator().manual_seed(5))

    def worker(owned: list[DrawingNetwork]) -> None:
        try:
            barrier.wait(timeout=30)
            for model in owned:
                convert(model, calibration_data=data)
        except BaseException as error:  # the main thread reports any worker failure
            failures.append(error)

    threads = [threading.Thread(target=worker, args=(owned,)) for owned in models]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not failures, failures
    assert not any(thread.is_alive() for thread in threads)
    assert_unchanged(before)


def test_a_failing_forward_still_restores_the_generators() -> None:
    """A forward that raises after drawing leaves no trace in the caller's generators."""
    model = DrawingNetwork(fail=True)
    random.seed(21)
    np.random.seed(22)
    torch.manual_seed(23)
    before = snapshot()
    with pytest.raises(ValueError, match="single-input tracing"):
        convert(model)
    assert_unchanged(before)


def test_activation_threshold_calibration_restores_the_generators() -> None:
    """The per-ReLU calibration runs the real model and still restores every generator."""
    model = DrawingNetwork()
    model.train()
    wrapped = nn.Sequential(model, nn.ReLU())
    random.seed(31)
    np.random.seed(32)
    torch.manual_seed(33)
    before = snapshot()
    scales = calibrate_activation_thresholds(
        wrapped,
        torch.rand(8, 3, generator=torch.Generator().manual_seed(6)),
    )
    assert_unchanged(before)
    assert len(scales) == 1 and scales[0] > 0
    assert model.training


def test_sections_nest_in_one_thread() -> None:
    """A section entered inside another restores its own states and the outer ones."""
    random.seed(41)
    before = snapshot()
    with preserved_random_state():
        random.random()
        with preserved_random_state():
            torch.rand(3)
        np.random.random_sample()
    assert_unchanged(before)
