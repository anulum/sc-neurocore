# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia parity of the contrastive SSL kernels

"""The Julia contrastive SSL lane matches the Python original within its stated bounds."""

from __future__ import annotations

from typing import Final

import numpy as np
import pytest

from sc_neurocore.contrastive.ssl import CSDPRule, SpikeContrastiveLoss
from tests.julia_requirement import require_julia

require_julia()

from sc_neurocore.accel import contrastive_ssl as dispatch

LANE: Final = "julia"


@pytest.fixture(autouse=True)
def _lane_is_ready() -> None:
    """Require the Julia lane to load once the bridge is present."""
    assert dispatch.backend_available(LANE)


@pytest.mark.parametrize("seed", range(40))
def test_infonce_loss_matches_python_within_the_stated_bound(seed: int) -> None:
    """Random batches, widths and temperatures agree within ``LOSS_RTOL``."""
    rng = np.random.default_rng(seed)
    batch = int(rng.integers(2, 33))
    features = int(rng.integers(1, 25))
    temperature = float(rng.uniform(0.05, 2.0))
    view_a = rng.normal(size=(batch, features))
    view_b = rng.normal(size=(batch, features))
    expected = SpikeContrastiveLoss(temperature).compute(view_a, view_b)
    actual = dispatch.infonce_loss(view_a, view_b, temperature=temperature, backend=LANE)
    assert actual == pytest.approx(expected, rel=dispatch.LOSS_RTOL[LANE], abs=0.0)


def test_infonce_edge_cases_match_python() -> None:
    """A single row, identical views, zero rows and saturating logits agree with the original."""
    rng = np.random.default_rng(11)
    view = rng.normal(size=(5, 3))
    cases = [
        (view[:1], view[:1], 0.5),
        (view, view, 0.5),
        (np.zeros((4, 3)), np.zeros((4, 3)), 0.5),
        (view * 1.0e6, view[::-1] * 1.0e6, 1.0e-3),
    ]
    for view_a, view_b, temperature in cases:
        expected = SpikeContrastiveLoss(temperature).compute(view_a, view_b)
        actual = dispatch.infonce_loss(view_a, view_b, temperature=temperature, backend=LANE)
        assert actual == pytest.approx(expected, rel=dispatch.LOSS_RTOL[LANE], abs=1.0e-12)


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("positive", [True, False])
def test_csdp_phases_match_python_within_the_stated_bound(seed: int, positive: bool) -> None:
    """Both CSDP phases agree within ``CSDP_TERM_EPS``, including zero-width weights."""
    rng = np.random.default_rng(seed)
    n_post = int(rng.integers(1, 12))
    n_pre = int(rng.integers(0, 12))
    lr = float(rng.uniform(0.0, 0.1))
    decay = float(rng.uniform(0.0, 0.01))
    weights = rng.normal(size=(n_post, n_pre))
    pre = rng.random(n_pre)
    post = rng.random(n_post)
    rule = CSDPRule(lr=lr, decay=decay)
    expected = (rule.positive_update if positive else rule.negative_update)(weights, pre, post)
    actual = dispatch.csdp_update(
        weights, pre, post, positive=positive, lr=lr, decay=decay, backend=LANE
    )
    assert actual.shape == expected.shape
    hebbian = lr * np.abs(np.outer(post, pre))
    scale = np.abs(weights) + hebbian + decay * np.abs(weights)
    bound = dispatch.CSDP_TERM_EPS[LANE] * np.finfo(np.float64).eps * scale
    assert np.all(np.abs(actual - expected) <= bound)


def test_goodness_matches_python_for_any_shape() -> None:
    """The goodness score agrees within ``GOODNESS_RTOL`` for vectors, matrices and empty input."""
    rng = np.random.default_rng(5)
    for values in (rng.normal(size=7), rng.normal(size=(3, 4)), rng.normal(size=4096), np.zeros(0)):
        expected = CSDPRule().goodness(values)
        actual = dispatch.goodness(values, backend=LANE)
        assert actual == pytest.approx(expected, rel=dispatch.GOODNESS_RTOL[LANE], abs=0.0)
