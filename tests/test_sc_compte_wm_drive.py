# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — deterministic SC Compte input receipts

"""Focused tests for the portable counter-addressed Poisson stream."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sc_neurocore.network import CounterPoissonDrive, SCCompteWMNetworkSpec


def test_counter_poisson_fixture_is_stable_and_receipted() -> None:
    drive = CounterPoissonDrive(64, 1800.0, 0.02, 42, 0)
    counts, receipt = drive.sample(0)
    assert np.flatnonzero(counts).tolist() == [49, 61]
    assert counts[counts > 0].tolist() == [1, 1]
    assert receipt.total_events == 2
    assert receipt.event_sha256 == (
        "f44a4e895cd9432f26b4ec05f67a7476e7b3a2d72043d477e9bc0395571b51cd"
    )


def test_counter_addressing_is_order_independent() -> None:
    drive = CounterPoissonDrive(2048, 1800.0, 0.02, 42, 0)
    later, later_receipt = drive.sample(1234)
    earlier, _ = drive.sample(7)
    repeated, repeated_receipt = drive.sample(1234)
    assert np.array_equal(later, repeated)
    assert later_receipt == repeated_receipt
    assert not np.array_equal(later, earlier)


def test_population_streams_and_seeds_are_separate() -> None:
    base = CounterPoissonDrive(512, 1800.0, 0.02, 42, 0)
    stream = CounterPoissonDrive(512, 1800.0, 0.02, 42, 1)
    seed = CounterPoissonDrive(512, 1800.0, 0.02, 43, 0)
    assert base.sample(9)[1].event_sha256 != stream.sample(9)[1].event_sha256
    assert base.sample(9)[1].event_sha256 != seed.sample(9)[1].event_sha256


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"population_size": 0}, "population_size"),
        ({"rate_hz": -1.0}, "rate_hz"),
        ({"dt_ms": 0.0}, "dt_ms"),
        ({"seed": -1}, "seed"),
        ({"stream": 1 << 64}, "stream"),
    ],
)
def test_counter_poisson_configuration_fails_closed(
    kwargs: dict[str, int | float], match: str
) -> None:
    values: dict[str, int | float] = {
        "population_size": 8,
        "rate_hz": 1800.0,
        "dt_ms": 0.02,
        "seed": 42,
        "stream": 0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        CounterPoissonDrive(
            population_size=int(values["population_size"]),
            rate_hz=float(values["rate_hz"]),
            dt_ms=float(values["dt_ms"]),
            seed=int(values["seed"]),
            stream=int(values["stream"]),
        )


def test_network_seed_contract_is_unsigned_64_bit() -> None:
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        replace(SCCompteWMNetworkSpec(), seed=1 << 64)
