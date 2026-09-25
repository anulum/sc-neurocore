# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Declared input encoders rebuild exactly from their declaration

"""Each encoder does what its declaration says, and the declaration rebuilds it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sc_neurocore.datasets import load_nmnist
from sc_neurocore.datasets.encoders import (
    ENCODER_SCHEMA,
    EventBinning,
    FirstSpikeLatency,
    InputEncoder,
    PoissonRates,
    encoder_from_declaration,
)
from tests.event_dataset_support import nmnist_event_bytes

#: A polarity mode the encoder does not define, typed loosely as a caller's JSON would be.
UNKNOWN: Any = "both"


def test_events_land_in_the_declared_step_and_channel() -> None:
    binning = EventBinning(dt_ms=1.0, n_steps=5, width=4, height=3)
    events = [
        [0, 0, 0, 0.0],  # step 0, OFF channel 0
        [3, 2, 1, 2.999],  # step 2, ON channel 12 + 11
        [1, 1, 0, 4.5],  # step 4, OFF channel 5
        [2, 0, 1, 5.0],  # at the end of the window: dropped
    ]
    spikes = binning.encode(np.array(events))
    assert spikes.shape == (5, 24) and spikes.dtype == bool
    assert sorted(zip(*np.nonzero(spikes), strict=True)) == [(0, 0), (2, 23), (4, 5)]


def test_merged_polarities_share_a_channel() -> None:
    binning = EventBinning(dt_ms=2.0, n_steps=2, width=2, height=2, polarity="merge")
    spikes = binning.encode(np.array([[1, 1, 0, 0.5], [1, 1, 1, 1.0], [0, 1, 1, 3.9]]))
    assert binning.channels == 4
    assert sorted(zip(*np.nonzero(spikes), strict=True)) == [(0, 3), (1, 2)]


def test_real_nmnist_events_bin_on_the_34_by_34_sensor(tmp_path: Path) -> None:
    folder = tmp_path / "Train" / "5"
    folder.mkdir(parents=True)
    (folder / "00000.bin").write_bytes(
        nmnist_event_bytes([(33, 33, 1, 250), (0, 17, 0, 9_999), (5, 5, 1, 50_000)])
    )
    (events,), _ = load_nmnist(tmp_path, train=True)
    spikes = EventBinning(dt_ms=5.0, n_steps=4, width=34, height=34).encode(events)
    assert sorted(zip(*np.nonzero(spikes), strict=True)) == [(0, 1156 + 33 * 34 + 33), (1, 17 * 34)]


def test_an_empty_recording_gives_an_empty_tensor() -> None:
    spikes = EventBinning(dt_ms=1.0, n_steps=3, width=2, height=2).encode(np.zeros((0, 4)))
    assert spikes.shape == (3, 8) and not spikes.any()


@pytest.mark.parametrize(
    ("events", "message"),
    [
        (np.zeros((2, 3)), r"shape \(N, 4\)"),
        (np.array([[4, 0, 0, 0.0]]), "outside the 4 x 3 sensor"),
        (np.array([[0, -1, 0, 0.0]]), "outside the 4 x 3 sensor"),
        (np.array([[0.5, 0, 0, 0.0]]), "whole numbers"),
        (np.array([[0, 0, 2, 0.0]]), "polarity must be 0 or 1"),
        (np.array([[0, 0, 0, -0.1]]), "negative time"),
        (np.array([[0, 0, 0, np.nan]]), "non-finite"),
    ],
)
def test_events_the_sensor_cannot_have_produced_are_refused(
    events: np.ndarray[Any, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        EventBinning(dt_ms=1.0, n_steps=5, width=4, height=3).encode(events)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: EventBinning(dt_ms=0.0, n_steps=1, width=1, height=1), "dt_ms must be positive"),
        (lambda: EventBinning(dt_ms=1.0, n_steps=0, width=1, height=1), "n_steps must be"),
        (lambda: EventBinning(dt_ms=1.0, n_steps=1, width=True, height=1), "width must be"),
        (lambda: EventBinning(dt_ms=1.0, n_steps=1, width=1, height=0), "height must be"),
        (
            lambda: EventBinning(dt_ms=1.0, n_steps=1, width=1, height=1, polarity=UNKNOWN),
            "polarity must be",
        ),
        (lambda: PoissonRates(n_steps=2, dt_ms=float("inf")), "dt_ms must be positive"),
        (lambda: FirstSpikeLatency(n_steps=2, tau=-1.0), "tau must be positive"),
    ],
)
def test_an_encoder_that_cannot_be_declared_is_refused(build: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build()


ENCODERS: list[InputEncoder] = [
    EventBinning(dt_ms=0.5, n_steps=8, width=34, height=34),
    EventBinning(dt_ms=1.0, n_steps=3, width=128, height=128, polarity="merge"),
    PoissonRates(n_steps=20, dt_ms=0.5, seed=11),
    FirstSpikeLatency(n_steps=10, tau=4.0),
]


@pytest.mark.parametrize("encoder", ENCODERS, ids=lambda encoder: type(encoder).__name__)
def test_a_declaration_rebuilds_the_identical_encoder(encoder: InputEncoder) -> None:
    declaration = json.loads(json.dumps(encoder.declaration()))
    rebuilt = encoder_from_declaration(declaration)
    assert rebuilt == encoder
    assert rebuilt.digest == encoder.digest
    assert declaration["schema"] == ENCODER_SCHEMA


def test_a_rebuilt_rate_encoder_draws_the_same_spikes() -> None:
    rates = np.linspace(0.0, 1.0, 16)
    encoder = PoissonRates(n_steps=50, dt_ms=0.5, seed=7)
    rebuilt = encoder_from_declaration(encoder.declaration())
    assert np.array_equal(rebuilt.encode(rates), encoder.encode(rates))
    # Probability per step is rate * dt_ms: a rate of 1 fires half the steps, roughly.
    assert 10 < int(encoder.encode(np.ones(1)).sum()) < 40


def test_latency_puts_larger_values_earlier_and_refuses_values_outside_the_unit_interval() -> None:
    encoder = FirstSpikeLatency(n_steps=10, tau=4.0)
    spikes = encoder.encode(np.array([1.0, 0.5, 0.0]))
    assert np.argmax(spikes, axis=0).tolist() == [0, 2, 4]
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        encoder.encode(np.array([1.5]))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        ({"schema": "other.v0"}, "is not 'sc-neurocore.input-encoder.v1'"),
        ({"encoder": "delta-modulation"}, "unknown encoder 'delta-modulation'"),
        ({"late_events": "clipped"}, "does not match what this version"),
    ],
)
def test_a_declaration_this_version_cannot_honour_is_refused(
    edit: dict[str, str], message: str
) -> None:
    declaration = EventBinning(dt_ms=1.0, n_steps=2, width=2, height=2).declaration()
    declaration.update(edit)
    with pytest.raises(ValueError, match=message):
        encoder_from_declaration(declaration)
