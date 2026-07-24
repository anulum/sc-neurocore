# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (load_formats) from former test_observation_loader.py

from __future__ import annotations

from observation_loader_support import *  # noqa: F403


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


def test_loads_vivado_text_reports_with_explicit_design_and_accuracy() -> None:
    observation = observation_from_synthesis_reports(
        {
            "utilisation": """
            +----------------------------+------+------+
            | Site Type                  | Used | Util |
            | CLB LUTs                   | 1,024| 10%  |
            +----------------------------+------+------+
            """,
            "power": """
            Power Report
            Total On-Chip Power (W): 0.125
            """,
        },
        design=_design(),
        accuracy_score=0.991,
        latency_cycles=128,
        source="vivado.rpt",
    )

    assert observation.luts_used == 1024
    assert observation.power_mw == 125.0
    assert observation.latency_cycles == 128
    assert observation.accuracy_score == 0.991


def test_loads_quartus_text_reports_and_embedded_latency() -> None:
    observation = observation_from_synthesis_reports(
        {
            "fit": """
            Fitter Resource Usage Summary
            ALMs needed : 118
            Latency (cycles): 96
            """,
            "power": """
            Power Analyzer Summary
            Total thermal power dissipation: 3.2 mW
            """,
        },
        design=_design(),
        accuracy_score=0.982,
        source="quartus.fit.rpt",
    )

    assert observation.luts_used == 118
    assert observation.power_mw == 3.2
    assert observation.latency_cycles == 96
    assert observation.accuracy_score == 0.982


def test_loads_synthesis_observation_from_report_files(tmp_path) -> None:
    utilisation = tmp_path / "utilisation.rpt"
    power = tmp_path / "power.rpt"
    utilisation.write_text("Slice LUTs | 421\nLatency: 64 cycles\n", encoding="utf-8")
    power.write_text("Total On-Chip Power (mW): 2.75\n", encoding="utf-8")

    observation = load_synthesis_observation(
        {"utilisation": utilisation, "power": power},
        design=_design(),
        accuracy_score=0.977,
    )

    assert observation.luts_used == 421
    assert observation.power_mw == 2.75
    assert observation.latency_cycles == 64
