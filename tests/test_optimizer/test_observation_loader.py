# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Optimiser observation loader tests

from __future__ import annotations

import json

import pytest

from sc_neurocore.optimizer import load_observations, observations_from_payload
from sc_neurocore.optimizer.observation_loader import ObservationLoadError
from sc_neurocore.optimizer.sc_optimizer import HardwareBudget, LayerProfile
from sc_neurocore.optimizer.surrogate_sc_optimizer import (
    SurrogateSCOptimizer,
    TargetHardwareProfile,
)


def _design() -> dict[str, object]:
    return {
        "mac_count": 256,
        "bitstream_length": 128,
        "decorrelator": "LFSR",
        "mode": "SC",
        "precision_bits": 8,
        "lfsr_polynomial": "x16+x15+x13+x4+1",
        "is_critical_path": True,
    }


def test_loads_generic_benchmark_records() -> None:
    payload = {
        "observations": [
            {
                **_design(),
                "luts_used": 320,
                "power_mw": 1.5,
                "latency_cycles": 128,
                "accuracy_score": 0.997,
            }
        ]
    }

    observations = observations_from_payload(payload)

    assert len(observations) == 1
    obs = observations[0]
    assert obs.mac_count == 256
    assert obs.luts_used == 320
    assert obs.power_mw == 1.5
    assert obs.accuracy_score == 0.997
    assert obs.is_critical_path is True


def test_loads_top_level_observation_list() -> None:
    observations = observations_from_payload(
        [
            {
                **_design(),
                "luts_used": 320,
                "power_mw": 1.5,
                "latency_cycles": 128,
                "accuracy_score": 0.997,
            }
        ]
    )

    assert len(observations) == 1
    assert observations[0].mac_count == 256


def test_loads_nested_candidate_and_measurement_views() -> None:
    payload = {
        "benchmark_observations": [
            {
                "candidate": _design(),
                "resources": {"logic_luts": 250},
                "power": {"total_power_mw": 1.6},
                "timing": {"latency": 64},
                "measurement": {"score": 0.975},
            }
        ]
    }

    observations = observations_from_payload(payload, source="nested.json")

    assert observations[0].luts_used == 250
    assert observations[0].power_mw == 1.6
    assert observations[0].latency_cycles == 64
    assert observations[0].accuracy_score == 0.975


def test_loads_vivado_style_manifest_with_design_defaults() -> None:
    payload = {
        "design_defaults": _design(),
        "observations": [
            {
                "report": {
                    "luts": 421,
                    "total_on_chip_power_mw": 2.75,
                    "latency_cycles": 128,
                    "accuracy": 0.991,
                }
            }
        ],
    }

    observations = observations_from_payload(payload, source="vivado.json")

    assert observations[0].luts_used == 421
    assert observations[0].power_mw == 2.75
    assert observations[0].accuracy_score == 0.991
    assert observations[0].lfsr_polynomial == "x16+x15+x13+x4+1"


def test_loads_quartus_aliases() -> None:
    payload = {
        **_design(),
        "measurement": {
            "alm": 118,
            "thermal_power_mw": 3.2,
            "cycles": 96,
            "parity_score": 0.982,
        },
    }

    observations = observations_from_payload(payload, source="quartus.json")

    assert observations[0].luts_used == 118
    assert observations[0].power_mw == 3.2
    assert observations[0].latency_cycles == 96
    assert observations[0].accuracy_score == 0.982


def test_loads_observations_from_file(tmp_path) -> None:
    path = tmp_path / "bench_observations.json"
    path.write_text(
        json.dumps(
            {
                "observations": [
                    {
                        **_design(),
                        "luts_used": 300,
                        "power_mw": 1.0,
                        "latency_cycles": 128,
                        "accuracy_score": 0.99,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    observations = load_observations(path)

    assert len(observations) == 1
    assert observations[0].luts_used == 300


def test_rejects_invalid_json_file(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ObservationLoadError, match="not valid JSON"):
        load_observations(path)


def test_rejects_payload_without_records() -> None:
    with pytest.raises(ObservationLoadError, match="contains no observation records"):
        observations_from_payload({"metadata": {"source": "empty"}}, source="empty.json")


def test_rejects_non_mapping_payload_and_record() -> None:
    with pytest.raises(ObservationLoadError, match="payload must be a JSON object or list"):
        observations_from_payload("bad")

    with pytest.raises(ObservationLoadError, match="record must be a JSON object"):
        observations_from_payload(["bad"])


def test_rejects_missing_measurement_fields() -> None:
    payload = {"observations": [{**_design(), "luts_used": 300}]}

    with pytest.raises(ObservationLoadError, match="missing one of power_mw"):
        observations_from_payload(payload, source="bad.json")


def test_rejects_non_numeric_measurement_values() -> None:
    payload = {
        "observations": [
            {
                **_design(),
                "luts_used": "not-a-number",
                "power_mw": 1.0,
                "latency_cycles": 128,
                "accuracy_score": 0.99,
            }
        ]
    }

    with pytest.raises(ObservationLoadError, match="luts_used must be an int"):
        observations_from_payload(payload, source="bad.json")


def test_rejects_invalid_design_and_negative_measurements() -> None:
    with pytest.raises(ObservationLoadError, match="decorrelator must be a string"):
        observations_from_payload(
            {
                "observations": [
                    {
                        **_design(),
                        "decorrelator": "",
                        "luts_used": 300,
                        "power_mw": 1.0,
                        "latency_cycles": 128,
                        "accuracy_score": 0.99,
                    }
                ]
            },
            source="bad.json",
        )

    with pytest.raises(ObservationLoadError, match="power_mw must be non-negative"):
        observations_from_payload(
            {
                "observations": [
                    {
                        **_design(),
                        "luts_used": 300,
                        "power_mw": -1.0,
                        "latency_cycles": 128,
                        "accuracy_score": 0.99,
                    }
                ]
            },
            source="bad.json",
        )


def test_loaded_observation_feeds_surrogate_optimizer() -> None:
    observations = observations_from_payload(
        {
            "observations": [
                {
                    **_design(),
                    "luts_used": 260,
                    "power_mw": 1.1,
                    "latency_cycles": 128,
                    "accuracy_score": 0.999,
                }
            ]
        }
    )
    target = TargetHardwareProfile(
        name="loader-integration",
        budget=HardwareBudget(max_luts=10_000, max_power_mw=100.0, max_latency_cycles=256),
    )
    optimiser = SurrogateSCOptimizer(
        target,
        bitstream_options=(64, 128),
        precision_options=(4, 8),
        observations=observations,
    )

    report = optimiser.optimise([LayerProfile("encoder", 256, is_critical_path=True)])

    assert report is not None
    assert report.feasible
    assert report.config["encoder"].bitstream_length == 128
    assert report.config["encoder"].precision_bits == 8
