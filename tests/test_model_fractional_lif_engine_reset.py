# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The engine's fractional LIF holds its reset as the Python model does

"""Compare the engine's fractional LIF with the Python model around a spike.

Both keep the potential after a reset in their memory. Before this was fixed, the
engine kept the value from before the reset, so the memory term undid every reset
and the engine fired on every step after its first spike, even under a drive below
threshold. The comparison stays within the first ``max_hist - 1`` steps: from step
``max_hist`` (counted from 0) the two truncate the Grünwald-Letnikov sum at different
lengths, since the engine uses all ``max_hist`` stored values and the Python model one
fewer.
"""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.models.fractional_lif import FractionalLIFNeuron

MAX_HIST = 50
#: A spike, a long sub-threshold stretch, a second spike, and a weaker stretch.
DRIVE = [1.5] + [0.5] * 30 + [1.2] + [0.3] * 15


@pytest.mark.parametrize("alpha", [0.5, 0.8, 1.0])
def test_the_engine_follows_the_python_model_through_a_reset(alpha: float) -> None:
    """Spike trains and potentials agree step by step while both memories have the same length."""
    engine = pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
    assert len(DRIVE) < MAX_HIST - 1
    rust = engine.FractionalLIFNeuron(alpha, MAX_HIST)
    python = FractionalLIFNeuron(alpha=alpha, _max_history=MAX_HIST)
    for index, current in enumerate(DRIVE):
        assert rust.step(current) == python.step(current), f"spike differs at step {index}"
        assert rust.get_state()["v"] == pytest.approx(python.v, rel=1e-12, abs=1e-15), (
            f"v differs at step {index}"
        )


def test_a_reset_holds_against_a_drive_below_threshold() -> None:
    """At alpha 1 the engine acts as a plain Euler LIF with dt = tau = 1: it settles at the drive."""
    engine = pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
    rust = engine.FractionalLIFNeuron(1.0, MAX_HIST)
    assert rust.step(1.5) == 1
    assert [rust.step(0.5) for _ in range(10)] == [0] * 10
    assert rust.get_state()["v"] == pytest.approx(0.5, abs=1e-12)
