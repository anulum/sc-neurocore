# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware experiment protocol and receipt acceptance

"""A hardware run counts as evidence only against the protocol declared before it.

Protocols bind a real converted network's digest and a real data digest; every
receipt is sealed, sent through JSON as an operator would store it and verified
again from its raw observations.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any

import numpy as np
import pytest

from sc_neurocore.conversion import ConvertedSNN
from sc_neurocore.conversion.loss_report import converted_sha256, data_sha256
from sc_neurocore.hardware.experiment import (
    PROTOCOL_SCHEMA_VERSION,
    RECEIPT_SCHEMA_VERSION,
    HardwareExperimentError,
    resolve_protocol,
    seal_receipt,
    verify_receipt,
)

_NETWORK = converted_sha256(ConvertedSNN([np.eye(3)], [None], [1.0], 8))
_DATA = data_sha256(np.full((4, 3), 0.5), np.array([0, 1, 2, 0]))


def _protocol(**change: Any) -> dict[str, Any]:
    """A complete protocol with energy measurement and all four criteria."""
    document: dict[str, Any] = {
        "opt_in": True,
        "declared_at": "2026-09-29T08:00:00+00:00",
        "operator": {"name": "Lab operator", "contact": "lab@example.org"},
        "device": {
            "vendor": "Lattice",
            "model": "iCE40 HX8K",
            "serial": "HX8K-0042",
            "firmware": "1.2",
        },
        "image": {"kind": "bitstream", "sha256": "a" * 64},
        "network_sha256": _NETWORK,
        "data_sha256": _DATA,
        "samples": 4,
        "latency": {
            "start_event": "host writes the request to the USB endpoint",
            "end_event": "host reads the last output byte",
            "clock": "host monotonic clock",
            "includes_transport": True,
            "warmup_runs": 2,
            "measured_runs": 5,
        },
        "power": {
            "instrument": {"vendor": "Keysight", "model": "N6705C", "serial": "MY-123"},
            "calibration": {
                "certificate": "CAL-2026-001",
                "calibrated_on": "2026-01-10",
                "valid_until": "2027-01-10",
            },
            "measurement_point": "VCCIO rail shunt",
            "sample_rate_hz": 50000,
        },
        "criteria": [
            {"metric": "accuracy", "threshold": 0.5},
            {"metric": "latency_p50_ms", "threshold": 3.0},
            {"metric": "latency_p95_ms", "threshold": 4.0},
            {"metric": "energy_per_inference_j", "threshold": 0.002},
        ],
    }
    document.update(change)
    return document


def _observations(**change: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "started_at": "2026-09-29T09:00:00+00:00",
        "finished_at": "2026-09-29T09:05:00+00:00",
        "device_serial": "HX8K-0042",
        "image_sha256": "a" * 64,
        "warmup_latency_ms": [9.0, 7.5],
        "latency_ms": [2.0, 2.5, 3.5, 2.2, 4.5],
        "correct": 3,
        "energy": {
            "source": "instrument",
            "instrument_serial": "MY-123",
            "joules_per_inference": [0.001, 0.0012, 0.0011, 0.001, 0.0016],
        },
        "notes": "room at 22 C",
    }
    document.update(change)
    return document


def _stored(document: dict[str, Any]) -> dict[str, Any]:
    """Round-trip through JSON as an operator's file would."""
    return json.loads(json.dumps(document))


class TestProtocol:
    def test_a_declared_protocol_binds_every_field(self) -> None:
        protocol = resolve_protocol(_protocol())
        public = protocol.to_public_dict()
        assert public["schema_version"] == PROTOCOL_SCHEMA_VERSION and public["opt_in"] is True
        assert public["network_sha256"] == _NETWORK and public["sha256"] == protocol.sha256
        assert resolve_protocol(_stored(public)) == protocol
        for path, value in [
            (("device", "serial"), "HX8K-0043"),
            (("image", "sha256"), "b" * 64),
            (("latency", "warmup_runs"), 3),
            (("power", "calibration", "certificate"), "CAL-2026-002"),
        ]:
            changed = copy.deepcopy(_protocol())
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            assert resolve_protocol(changed).sha256 != protocol.sha256
        with pytest.raises(HardwareExperimentError, match="digest does not match"):
            resolve_protocol({**public, "samples": 5})

    def test_a_protocol_without_power_and_criteria_is_admitted(self) -> None:
        protocol = resolve_protocol(_protocol(power=None, criteria=[]))
        assert protocol.power is None and protocol.criteria == []
        assert resolve_protocol(
            {k: v for k, v in _protocol().items() if k not in {"power", "criteria"}}
        )

    @pytest.mark.parametrize(
        "change,field",
        [
            ({"opt_in": False}, "opt_in"),
            ({"opt_in": "yes"}, "opt_in"),
            ({"schema_version": "v0"}, "schema_version"),
            ({"surprise": 1}, "protocol"),
            ({"declared_at": "yesterday"}, "declared_at"),
            ({"declared_at": "2026-09-29T08:00:00"}, "declared_at"),
            ({"operator": {"name": " "}}, "operator.name"),
            ({"operator": {"name": "x", "badge": 1}}, "operator"),
            ({"operator": "someone"}, "operator"),
            ({"device": {"vendor": "L", "model": "M"}}, "device.serial"),
            ({"image": {"kind": "wheel", "sha256": "a" * 64}}, "image.kind"),
            ({"image": {"kind": "bitstream", "sha256": "A" * 64}}, "image.sha256"),
            ({"image": {"kind": "bitstream", "sha256": "a" * 63}}, "image.sha256"),
            ({"network_sha256": 5}, "network_sha256"),
            ({"samples": 0}, "samples"),
            ({"samples": True}, "samples"),
            ({"power": "a meter"}, "power"),
            ({"criteria": "accuracy"}, "criteria"),
            ({"criteria": [{"metric": "throughput", "threshold": 1}]}, "criteria[0].metric"),
            ({"criteria": [{"metric": "accuracy", "threshold": 1.5}]}, "criteria[0].threshold"),
            (
                {"criteria": [{"metric": "latency_p50_ms", "threshold": -1}]},
                "criteria[0].threshold",
            ),
            (
                {"criteria": [{"metric": "latency_p50_ms", "threshold": math.inf}]},
                "criteria[0].threshold",
            ),
            ({"criteria": [{"metric": "accuracy", "threshold": True}]}, "criteria[0].threshold"),
            ({"criteria": [{"metric": "accuracy", "threshold": 1, "why": ""}]}, "criteria[0]"),
            (
                {"power": None, "criteria": [{"metric": "energy_per_inference_j", "threshold": 1}]},
                "criteria[0].metric",
            ),
        ],
    )
    def test_a_protocol_that_could_not_be_evidence_is_refused(
        self, change: dict[str, Any], field: str
    ) -> None:
        with pytest.raises(HardwareExperimentError) as refusal:
            resolve_protocol(_protocol(**change))
        assert refusal.value.field == field

    @pytest.mark.parametrize(
        "latency_change,field",
        [
            ({"includes_transport": False}, "latency.includes_transport"),
            ({"includes_transport": None}, "latency.includes_transport"),
            ({"warmup_runs": -1}, "latency.warmup_runs"),
            ({"measured_runs": 0}, "latency.measured_runs"),
            ({"clock": ""}, "latency.clock"),
            ({"jitter": 1}, "latency"),
        ],
    )
    def test_latency_must_include_transport_and_declare_its_runs(
        self, latency_change: dict[str, Any], field: str
    ) -> None:
        latency = {**_protocol()["latency"], **latency_change}
        with pytest.raises(HardwareExperimentError) as refusal:
            resolve_protocol(_protocol(latency=latency))
        assert refusal.value.field == field

    @pytest.mark.parametrize(
        "power_change,field",
        [
            ({"instrument": {"vendor": "K", "model": "N"}}, "power.instrument.serial"),
            (
                {"instrument": {"vendor": "K", "model": "N", "serial": "S", "x": 1}},
                "power.instrument",
            ),
            (
                {"calibration": {"certificate": "C", "calibrated_on": "2026-01-10"}},
                "power.calibration.valid_until",
            ),
            (
                {
                    "calibration": {
                        "certificate": "C",
                        "calibrated_on": "2026-02-01",
                        "valid_until": "2026-01-01",
                    }
                },
                "power.calibration.valid_until",
            ),
            (
                {
                    "calibration": {
                        "certificate": "",
                        "calibrated_on": "2026-01-01",
                        "valid_until": "2027-01-01",
                    }
                },
                "power.calibration.certificate",
            ),
            (
                {
                    "calibration": {
                        "certificate": "C",
                        "calibrated_on": "01/01/2026",
                        "valid_until": "2027-01-01",
                    }
                },
                "power.calibration.calibrated_on",
            ),
            (
                {
                    "calibration": {
                        "certificate": "C",
                        "calibrated_on": "2026-01-01",
                        "valid_until": "2027-01-01",
                        "lab": 1,
                    }
                },
                "power.calibration",
            ),
            ({"sample_rate_hz": 0}, "power.sample_rate_hz"),
            ({"sample_rate_hz": True}, "power.sample_rate_hz"),
            ({"sample_rate_hz": math.inf}, "power.sample_rate_hz"),
            ({"measurement_point": ""}, "power.measurement_point"),
            ({"meter": "x"}, "power"),
        ],
    )
    def test_a_power_declaration_needs_an_identified_calibrated_instrument(
        self, power_change: dict[str, Any], field: str
    ) -> None:
        power = {**_protocol()["power"], **power_change}
        with pytest.raises(HardwareExperimentError) as refusal:
            resolve_protocol(_protocol(power=power))
        assert refusal.value.field == field


class TestReceipt:
    def test_a_sealed_receipt_recomputes_from_its_observations(self) -> None:
        protocol = resolve_protocol(_protocol())
        receipt = seal_receipt(protocol, _observations())
        assert receipt["schema_version"] == RECEIPT_SCHEMA_VERSION
        assert receipt["protocol_sha256"] == protocol.sha256
        derived = receipt["derived"]
        assert derived["accuracy"] == 0.75
        assert derived["latency_p50_ms"] == 2.5 and derived["latency_p95_ms"] == 4.5
        assert derived["latency_percentile_method"] == "nearest-rank"
        assert derived["energy_per_inference_j"] == pytest.approx(0.00118)
        assert receipt["observations"]["energy"]["max_j"] == 0.0016
        assert receipt["observations"]["warmup_latency_ms"] == [9.0, 7.5]
        verdicts = {verdict["metric"]: verdict["passed"] for verdict in receipt["verdicts"]}
        assert verdicts == {
            "accuracy": True,
            "latency_p50_ms": True,
            "latency_p95_ms": False,
            "energy_per_inference_j": True,
        }
        assert verify_receipt(_stored(receipt)) == receipt

    def test_optional_text_may_be_left_out(self) -> None:
        protocol = resolve_protocol(
            _protocol(
                operator={"name": "Lab operator"},
                device={"vendor": "Lattice", "model": "iCE40 HX8K", "serial": "HX8K-0042"},
            )
        )
        assert protocol.operator["contact"] == "" and protocol.device["firmware"] == ""
        observations = _observations()
        del observations["notes"]
        assert seal_receipt(protocol, observations)["observations"]["notes"] == ""

    def test_a_run_without_power_has_no_energy_figure(self) -> None:
        protocol = resolve_protocol(
            _protocol(power=None, criteria=[{"metric": "accuracy", "threshold": 1}])
        )
        receipt = seal_receipt(protocol, _observations(energy=None))
        assert receipt["derived"]["energy_per_inference_j"] is None
        assert receipt["verdicts"][0]["passed"] is False
        assert verify_receipt(_stored(receipt))["sha256"] == receipt["sha256"]

    @pytest.mark.parametrize(
        "change,field",
        [
            ({"device_serial": "HX8K-9999"}, "device_serial"),
            ({"image_sha256": "b" * 64}, "image_sha256"),
            ({"started_at": "2026-09-29T07:59:59+00:00"}, "started_at"),
            ({"finished_at": "2026-09-29T09:00:00+00:00"}, "finished_at"),
            ({"finished_at": "soon"}, "finished_at"),
            ({"warmup_latency_ms": [9.0]}, "warmup_latency_ms"),
            ({"latency_ms": [2.0, 2.5, 3.5, 2.2]}, "latency_ms"),
            ({"latency_ms": "2.0"}, "latency_ms"),
            ({"latency_ms": [2.0, 2.5, 3.5, 2.2, -1]}, "latency_ms"),
            ({"latency_ms": [2.0, 2.5, 3.5, 2.2, math.nan]}, "latency_ms"),
            ({"correct": 5}, "correct"),
            ({"correct": -1}, "correct"),
            ({"energy": None}, "energy"),
            (
                {
                    "energy": {
                        "source": "operation_count",
                        "instrument_serial": "MY-123",
                        "joules_per_inference": [0.001] * 5,
                    }
                },
                "energy.source",
            ),
            (
                {
                    "energy": {
                        "source": "estimate",
                        "instrument_serial": "MY-123",
                        "joules_per_inference": [0.001] * 5,
                    }
                },
                "energy.source",
            ),
            (
                {
                    "energy": {
                        "source": "instrument",
                        "instrument_serial": "OTHER",
                        "joules_per_inference": [0.001] * 5,
                    }
                },
                "energy.instrument_serial",
            ),
            (
                {
                    "energy": {
                        "source": "instrument",
                        "instrument_serial": "MY-123",
                        "joules_per_inference": [0.001] * 4,
                    }
                },
                "energy.joules_per_inference",
            ),
            (
                {
                    "energy": {
                        "source": "instrument",
                        "instrument_serial": "MY-123",
                        "joules_per_inference": [0.001] * 5,
                        "ops": 1,
                    }
                },
                "energy",
            ),
            ({"energy": "1 mJ"}, "energy"),
            ({"operator_says": "fine"}, "observations"),
        ],
    )
    def test_observations_that_do_not_belong_to_the_protocol_are_refused(
        self, change: dict[str, Any], field: str
    ) -> None:
        with pytest.raises(HardwareExperimentError) as refusal:
            seal_receipt(resolve_protocol(_protocol()), _observations(**change))
        assert refusal.value.field == field

    def test_energy_is_refused_when_no_instrument_was_declared(self) -> None:
        protocol = resolve_protocol(_protocol(power=None, criteria=[]))
        with pytest.raises(HardwareExperimentError, match="declared no power instrument"):
            seal_receipt(protocol, _observations())

    @pytest.mark.parametrize("day", ["2026-01-09", "2027-01-11"])
    def test_energy_from_an_expired_or_future_calibration_is_refused(self, day: str) -> None:
        protocol = resolve_protocol(_protocol(declared_at=f"{day}T00:00:00+00:00"))
        with pytest.raises(HardwareExperimentError, match="calibration was not valid"):
            seal_receipt(
                protocol,
                _observations(
                    started_at=f"{day}T09:00:00+00:00", finished_at=f"{day}T09:05:00+00:00"
                ),
            )

    @pytest.mark.parametrize(
        "path,value,field",
        [
            (("derived", "accuracy"), 1.0, "receipt"),
            (("verdicts",), [], "receipt"),
            (("observations", "latency_ms"), [1.0, 1.0, 1.0, 1.0, 1.0], "receipt"),
            (("sha256",), "0" * 64, "receipt"),
            (("protocol_sha256",), "0" * 64, "protocol_sha256"),
            (("schema_version",), "v0", "schema_version"),
            (("protocol", "samples"), 8, "sha256"),
        ],
    )
    def test_a_changed_receipt_does_not_verify(
        self, path: tuple[str, ...], value: Any, field: str
    ) -> None:
        receipt = _stored(seal_receipt(resolve_protocol(_protocol()), _observations()))
        target = receipt
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(HardwareExperimentError) as refusal:
            verify_receipt(receipt)
        assert refusal.value.field == field

    def test_a_non_object_receipt_is_refused(self) -> None:
        with pytest.raises(HardwareExperimentError, match="must be an object"):
            verify_receipt([])
