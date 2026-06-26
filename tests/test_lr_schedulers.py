# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for learning rate schedulers

"""Tests for learning rate schedulers."""

from __future__ import annotations

import math

from sc_neurocore.learning.schedulers import (
    StepScheduler,
    ExponentialScheduler,
    CosineScheduler,
    WarmupCosineScheduler,
)


def test_step_scheduler_drops_at_interval() -> None:
    s = StepScheduler(lr_init=0.1, step_size=5, gamma=0.5)
    for _ in range(4):
        s.step()
    assert s.lr == 0.1
    s.step()  # step 5
    assert abs(s.lr - 0.05) < 1e-10


def test_step_scheduler_multiple_drops() -> None:
    s = StepScheduler(lr_init=1.0, step_size=3, gamma=0.1)
    for _ in range(6):
        s.step()
    assert abs(s.lr - 0.01) < 1e-10


def test_exponential_scheduler_decays() -> None:
    s = ExponentialScheduler(lr_init=1.0, gamma=0.9)
    for _ in range(10):
        s.step()
    assert abs(s.lr - 0.9**10) < 1e-10


def test_cosine_scheduler_endpoints() -> None:
    s = CosineScheduler(lr_init=0.1, lr_min=0.001, total_steps=100)
    assert s.lr == 0.1
    for _ in range(100):
        s.step()
    assert abs(s.lr - 0.001) < 1e-6


def test_cosine_scheduler_midpoint() -> None:
    s = CosineScheduler(lr_init=1.0, lr_min=0.0, total_steps=100)
    for _ in range(50):
        s.step()
    expected = 0.5 * (1 + math.cos(math.pi * 0.5))
    assert abs(s.lr - expected) < 1e-6


def test_warmup_cosine_warmup_phase() -> None:
    s = WarmupCosineScheduler(lr_init=0.1, lr_min=0.001, warmup_steps=10, total_steps=100)
    assert s.lr == 0.0
    for _ in range(5):
        s.step()
    assert abs(s.lr - 0.05) < 1e-10


def test_warmup_cosine_peak_and_decay() -> None:
    s = WarmupCosineScheduler(lr_init=0.1, lr_min=0.001, warmup_steps=10, total_steps=110)
    for _ in range(10):
        s.step()
    assert abs(s.lr - 0.1) < 1e-10
    for _ in range(100):
        s.step()
    assert abs(s.lr - 0.001) < 1e-4


def test_cosine_reset() -> None:
    s = CosineScheduler(lr_init=0.1, lr_min=0.0, total_steps=50)
    for _ in range(50):
        s.step()
    s.reset()
    assert s.lr == 0.1
    assert s._count == 0


def test_step_scheduler_reset() -> None:
    s = StepScheduler(lr_init=1.0, step_size=3, gamma=0.1)
    for _ in range(6):
        s.step()
    s.reset()
    assert s._count == 0


def test_exponential_scheduler_reset() -> None:
    s = ExponentialScheduler(lr_init=1.0, gamma=0.9)
    for _ in range(10):
        s.step()
    s.reset()


def test_warmup_cosine_reset() -> None:
    s = WarmupCosineScheduler(lr_init=0.1, lr_min=0.001, warmup_steps=10, total_steps=100)
    for _ in range(50):
        s.step()
    s.reset()
    assert s.lr == 0.0
    assert s._count == 0
