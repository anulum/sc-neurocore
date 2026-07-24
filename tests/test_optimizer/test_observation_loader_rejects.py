# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (rejects) from former test_observation_loader.py

from __future__ import annotations

from observation_loader_support import *  # noqa: F403

def test_rejects_incomplete_synthesis_reports_without_fabricating_metrics() -> None:
    with pytest.raises(ObservationLoadError, match="missing power_mw"):
        observation_from_synthesis_reports(
            {"utilisation": "CLB LUTs | 256"},
            design=_design(),
            accuracy_score=0.99,
            latency_cycles=32,
            source="missing-power.rpt",
        )

    with pytest.raises(ObservationLoadError, match="missing one of latency_cycles"):
        observation_from_synthesis_reports(
            {
                "utilisation": "CLB LUTs | 256",
                "power": "Total On-Chip Power (W): 0.01",
            },
            design=_design(),
            accuracy_score=0.99,
            source="missing-latency.rpt",
        )


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


def test_rejects_boolean_integer_fields() -> None:
    payload = {
        "observations": [
            {
                **_design(),
                "luts_used": True,
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


def test_rejects_non_finite_float_measurements() -> None:
    payload = {
        "observations": [
            {
                **_design(),
                "luts_used": 300,
                "power_mw": float("nan"),
                "latency_cycles": 128,
                "accuracy_score": 0.99,
            }
        ]
    }

    with pytest.raises(ObservationLoadError, match="power_mw must be finite"):
        observations_from_payload(payload, source="bad.json")
