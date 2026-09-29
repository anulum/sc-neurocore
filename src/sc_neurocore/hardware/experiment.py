# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware experiment protocol and receipt

"""Declare a hardware experiment before it runs, then seal what it measured.

A **protocol** is written before any execution. It records the operator's
explicit opt-in, who runs it, on which device, with which image, which
converted network and data, how latency is timed (transport included, warmup
runs declared and excluded), how power is measured (instrument and a valid
calibration certificate) and the preregistered acceptance criteria. Its digest
binds all of that.

A **receipt** binds observations to that digest. It refuses observations from
another device or image, run counts that differ from the protocol, a run that
started before the protocol was declared, energy from an undeclared or
uncalibrated instrument, and any energy figure not measured by an instrument:
energy is never inferred from operation counts or estimates. Accuracy, latency
percentiles, energy statistics and every verdict are computed here from the
raw observations, never taken from the caller, so :func:`verify_receipt` can
recompute a sealed receipt end to end.

Nothing here executes on hardware. Execution is the operator's act on the
operator's device; this module states what counts as evidence of it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

PROTOCOL_SCHEMA_VERSION = "sc-neurocore.hardware-experiment-protocol.v1"
RECEIPT_SCHEMA_VERSION = "sc-neurocore.hardware-experiment-receipt.v1"

#: Judged metrics and the direction in which each passes.
CRITERIA: Mapping[str, str] = {
    "accuracy": "at_least",
    "latency_p50_ms": "at_most",
    "latency_p95_ms": "at_most",
    "energy_per_inference_j": "at_most",
}

#: What an executed image may be.
IMAGE_KINDS: tuple[str, ...] = ("bitstream", "firmware", "container", "binary")

_ENERGY_SOURCE = "instrument"


class HardwareExperimentError(ValueError):
    """Raised when a protocol or its observations cannot be admitted as evidence.

    Attributes
    ----------
    field : str
        The protocol or observation field that was refused.
    reason : str
        What is wrong, in words an operator can act on.
    """

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


def _canonical_sha256(document: Mapping[str, Any]) -> str:
    """Digest a document's canonical JSON form."""
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HardwareExperimentError(field, "must be an object.")
    return value


def _text(value: object, field: str, *, required: bool = True) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()):
        raise HardwareExperimentError(field, "must be non-empty text.")
    return value


def _sha256_hex(value: object, field: str) -> str:
    text = _text(value, field)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise HardwareExperimentError(field, "must be 64 lowercase hexadecimal characters.")
    return text


def _count(value: object, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HardwareExperimentError(field, f"must be an integer of at least {minimum}.")
    return value


def _instant(value: object, field: str) -> datetime:
    text = _text(value, field)
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise HardwareExperimentError(field, "must be an ISO 8601 timestamp.") from exc
    if moment.tzinfo is None:
        raise HardwareExperimentError(field, "must carry a time zone.")
    return moment


def _day(value: object, field: str) -> date:
    text = _text(value, field)
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise HardwareExperimentError(field, "must be an ISO 8601 date.") from exc


def _unknown(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    extra = sorted(str(key) for key in set(value) - allowed)
    if extra:
        raise HardwareExperimentError(field, f"unknown field(s) {', '.join(extra)}.")


@dataclass(frozen=True)
class Protocol:
    """A hardware experiment as declared before it runs.

    Attributes
    ----------
    declared_at : str
        ISO 8601 timestamp with zone at which the protocol was fixed.
    operator : dict
        ``name`` and optional ``contact`` of whoever authorised and runs it.
    device : dict
        ``vendor``, ``model``, ``serial`` and optional ``firmware``.
    image : dict
        ``kind`` (one of :data:`IMAGE_KINDS`) and ``sha256`` of what executes.
    network_sha256, data_sha256 : str
        Digests of the converted network and of the evaluation data.
    samples : int
        Evaluation samples per measured run.
    latency : dict
        ``start_event``, ``end_event``, ``clock``, ``warmup_runs`` and
        ``measured_runs``; ``includes_transport`` is always true.
    power : dict or None
        ``instrument`` (vendor, model, serial), ``calibration`` (certificate,
        calibrated_on, valid_until), ``measurement_point`` and
        ``sample_rate_hz``; ``None`` when energy is not measured.
    criteria : list of dict
        Preregistered ``metric`` and ``threshold`` pairs.
    """

    declared_at: str
    operator: dict[str, str]
    device: dict[str, str]
    image: dict[str, str]
    network_sha256: str
    data_sha256: str
    samples: int
    latency: dict[str, Any]
    power: dict[str, Any] | None
    criteria: list[dict[str, Any]]

    def to_public_dict(self) -> dict[str, Any]:
        """Return the protocol as stored, including its opt-in and digest.

        Returns
        -------
        dict
            Every field, ``opt_in: true``, the schema version and ``sha256``.
        """
        body = {"schema_version": PROTOCOL_SCHEMA_VERSION, "opt_in": True, **asdict(self)}
        return {**body, "sha256": _canonical_sha256(body)}

    @property
    def sha256(self) -> str:
        """Return the digest that a receipt binds to."""
        return str(self.to_public_dict()["sha256"])


def _latency(value: object) -> dict[str, Any]:
    data = _mapping(value, "latency")
    _unknown(
        data,
        {"start_event", "end_event", "clock", "includes_transport", "warmup_runs", "measured_runs"},
        "latency",
    )
    if data.get("includes_transport") is not True:
        raise HardwareExperimentError(
            "latency.includes_transport",
            "latency is measured from the host request to the host receiving the result, "
            "transport included; declare includes_transport: true.",
        )
    return {
        "start_event": _text(data.get("start_event"), "latency.start_event"),
        "end_event": _text(data.get("end_event"), "latency.end_event"),
        "clock": _text(data.get("clock"), "latency.clock"),
        "includes_transport": True,
        "warmup_runs": _count(data.get("warmup_runs"), "latency.warmup_runs", minimum=0),
        "measured_runs": _count(data.get("measured_runs"), "latency.measured_runs", minimum=1),
    }


def _power(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    data = _mapping(value, "power")
    _unknown(data, {"instrument", "calibration", "measurement_point", "sample_rate_hz"}, "power")
    instrument = _mapping(data.get("instrument"), "power.instrument")
    _unknown(instrument, {"vendor", "model", "serial"}, "power.instrument")
    calibration = _mapping(data.get("calibration"), "power.calibration")
    _unknown(calibration, {"certificate", "calibrated_on", "valid_until"}, "power.calibration")
    calibrated_on = _day(calibration.get("calibrated_on"), "power.calibration.calibrated_on")
    valid_until = _day(calibration.get("valid_until"), "power.calibration.valid_until")
    if valid_until < calibrated_on:
        raise HardwareExperimentError(
            "power.calibration.valid_until", "must not precede the calibration date."
        )
    rate = data.get("sample_rate_hz")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not rate > 0:
        raise HardwareExperimentError("power.sample_rate_hz", "must be a positive number.")
    if not math.isfinite(float(rate)):
        raise HardwareExperimentError("power.sample_rate_hz", "must be a positive number.")
    return {
        "instrument": {
            key: _text(instrument.get(key), f"power.instrument.{key}")
            for key in ("vendor", "model", "serial")
        },
        "calibration": {
            "certificate": _text(calibration.get("certificate"), "power.calibration.certificate"),
            "calibrated_on": calibrated_on.isoformat(),
            "valid_until": valid_until.isoformat(),
        },
        "measurement_point": _text(data.get("measurement_point"), "power.measurement_point"),
        "sample_rate_hz": float(rate),
    }


def _criteria(value: object, power: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise HardwareExperimentError("criteria", "must be a list of preregistered criteria.")
    resolved: list[dict[str, Any]] = []
    for index, entry in enumerate(value):
        field = f"criteria[{index}]"
        data = _mapping(entry, field)
        _unknown(data, {"metric", "threshold"}, field)
        metric = data.get("metric")
        if metric not in CRITERIA:
            raise HardwareExperimentError(
                f"{field}.metric", f"must be one of {', '.join(CRITERIA)}."
            )
        threshold = data.get("threshold")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or threshold < 0
            or (metric == "accuracy" and threshold > 1)
        ):
            raise HardwareExperimentError(
                f"{field}.threshold", "must be finite and non-negative, at most 1 for accuracy."
            )
        if metric == "energy_per_inference_j" and power is None:
            raise HardwareExperimentError(
                f"{field}.metric", "an energy criterion needs a declared power instrument."
            )
        resolved.append({"metric": metric, "threshold": float(threshold)})
    return resolved


def resolve_protocol(value: object) -> Protocol:
    """Resolve a declared protocol, refusing anything that could not be evidence.

    Parameters
    ----------
    value : mapping
        The protocol document. ``opt_in`` must be ``true``; a stored protocol
        resubmitted with its ``sha256`` must match it.

    Returns
    -------
    Protocol
        The protocol exactly as a receipt will bind it.

    Raises
    ------
    HardwareExperimentError
        A missing opt-in or identity, an unknown field, a latency definition
        without transport, a power declaration without a valid calibration, or
        a criterion that cannot be judged.
    """
    data = _mapping(value, "protocol")
    _unknown(
        data,
        {
            "schema_version",
            "opt_in",
            "declared_at",
            "operator",
            "device",
            "image",
            "network_sha256",
            "data_sha256",
            "samples",
            "latency",
            "power",
            "criteria",
            "sha256",
        },
        "protocol",
    )
    if data.get("schema_version", PROTOCOL_SCHEMA_VERSION) != PROTOCOL_SCHEMA_VERSION:
        raise HardwareExperimentError("schema_version", "is not the protocol this build reads.")
    if data.get("opt_in") is not True:
        raise HardwareExperimentError(
            "opt_in", "hardware execution runs only on the operator's explicit opt_in: true."
        )
    operator = _mapping(data.get("operator"), "operator")
    _unknown(operator, {"name", "contact"}, "operator")
    device = _mapping(data.get("device"), "device")
    _unknown(device, {"vendor", "model", "serial", "firmware"}, "device")
    image = _mapping(data.get("image"), "image")
    _unknown(image, {"kind", "sha256"}, "image")
    if image.get("kind") not in IMAGE_KINDS:
        raise HardwareExperimentError("image.kind", f"must be one of {', '.join(IMAGE_KINDS)}.")
    power = _power(data.get("power"))
    protocol = Protocol(
        declared_at=_instant(data.get("declared_at"), "declared_at").isoformat(),
        operator={
            "name": _text(operator.get("name"), "operator.name"),
            "contact": _text(operator.get("contact"), "operator.contact", required=False),
        },
        device={
            "vendor": _text(device.get("vendor"), "device.vendor"),
            "model": _text(device.get("model"), "device.model"),
            "serial": _text(device.get("serial"), "device.serial"),
            "firmware": _text(device.get("firmware"), "device.firmware", required=False),
        },
        image={
            "kind": str(image["kind"]),
            "sha256": _sha256_hex(image.get("sha256"), "image.sha256"),
        },
        network_sha256=_sha256_hex(data.get("network_sha256"), "network_sha256"),
        data_sha256=_sha256_hex(data.get("data_sha256"), "data_sha256"),
        samples=_count(data.get("samples"), "samples", minimum=1),
        latency=_latency(data.get("latency")),
        power=power,
        criteria=_criteria(data.get("criteria", []), power),
    )
    declared = data.get("sha256")
    if declared is not None and declared != protocol.sha256:
        raise HardwareExperimentError("sha256", "the stored protocol digest does not match it.")
    return protocol


def _durations(value: object, field: str, runs: int) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != runs:
        raise HardwareExperimentError(field, f"must hold exactly {runs} measurement(s).")
    result: list[float] = []
    for entry in value:
        if (
            isinstance(entry, bool)
            or not isinstance(entry, (int, float))
            or not math.isfinite(float(entry))
            or entry < 0
        ):
            raise HardwareExperimentError(
                field, "every measurement must be finite and non-negative."
            )
        result.append(float(entry))
    return result


def _nearest_rank(values: list[float], quantile: float) -> float:
    """Return the nearest-rank percentile: the ceil(q*n)-th smallest value."""
    ordered = sorted(values)
    return ordered[max(math.ceil(quantile * len(ordered)), 1) - 1]


def _energy(protocol: Protocol, value: object, finished: datetime) -> dict[str, Any] | None:
    power = protocol.power
    if value is None:
        if power is not None:
            raise HardwareExperimentError(
                "energy", "the protocol declared a power instrument; its measurement is missing."
            )
        return None
    data = _mapping(value, "energy")
    _unknown(data, {"source", "instrument_serial", "joules_per_inference"}, "energy")
    if data.get("source") != _ENERGY_SOURCE:
        raise HardwareExperimentError(
            "energy.source",
            "energy is measured by the declared instrument; it is never inferred from "
            "operation counts or estimates.",
        )
    if power is None:
        raise HardwareExperimentError("energy", "the protocol declared no power instrument.")
    if data.get("instrument_serial") != power["instrument"]["serial"]:
        raise HardwareExperimentError(
            "energy.instrument_serial", "is not the instrument the protocol declared."
        )
    calibration = power["calibration"]
    day = finished.date()
    if (
        not date.fromisoformat(calibration["calibrated_on"])
        <= day
        <= date.fromisoformat(calibration["valid_until"])
    ):
        raise HardwareExperimentError(
            "energy", "the instrument's calibration was not valid on the day of the run."
        )
    joules = _durations(
        data.get("joules_per_inference"),
        "energy.joules_per_inference",
        protocol.latency["measured_runs"],
    )
    return {
        "source": _ENERGY_SOURCE,
        "instrument_serial": power["instrument"]["serial"],
        "joules_per_inference": joules,
        "mean_j": sum(joules) / len(joules),
        "max_j": max(joules),
    }


def _judge(criterion: Mapping[str, Any], observed: float | None) -> dict[str, Any]:
    metric = str(criterion["metric"])
    direction = CRITERIA[metric]
    threshold = float(criterion["threshold"])
    passed = observed is not None and (
        observed >= threshold if direction == "at_least" else observed <= threshold
    )
    return {
        "metric": metric,
        "direction": direction,
        "threshold": threshold,
        "observed": observed,
        "passed": passed,
    }


def _derive(protocol: Protocol, observations: Mapping[str, Any]) -> dict[str, Any]:
    """Admit raw observations against the protocol and compute every derived figure."""
    _unknown(
        observations,
        {
            "started_at",
            "finished_at",
            "device_serial",
            "image_sha256",
            "warmup_latency_ms",
            "latency_ms",
            "correct",
            "energy",
            "notes",
        },
        "observations",
    )
    started = _instant(observations.get("started_at"), "started_at")
    finished = _instant(observations.get("finished_at"), "finished_at")
    if started < datetime.fromisoformat(protocol.declared_at):
        raise HardwareExperimentError("started_at", "precedes the protocol's declaration.")
    if finished <= started:
        raise HardwareExperimentError("finished_at", "must follow started_at.")
    if observations.get("device_serial") != protocol.device["serial"]:
        raise HardwareExperimentError("device_serial", "is not the device the protocol declared.")
    if observations.get("image_sha256") != protocol.image["sha256"]:
        raise HardwareExperimentError("image_sha256", "is not the image the protocol declared.")
    latency = protocol.latency
    warmup = _durations(
        observations.get("warmup_latency_ms"), "warmup_latency_ms", latency["warmup_runs"]
    )
    measured = _durations(observations.get("latency_ms"), "latency_ms", latency["measured_runs"])
    correct = _count(observations.get("correct"), "correct", minimum=0)
    if correct > protocol.samples:
        raise HardwareExperimentError("correct", "exceeds the protocol's sample count.")
    energy = _energy(protocol, observations.get("energy"), finished)
    derived = {
        "accuracy": correct / protocol.samples,
        "latency_p50_ms": _nearest_rank(measured, 0.50),
        "latency_p95_ms": _nearest_rank(measured, 0.95),
        "energy_per_inference_j": None if energy is None else energy["mean_j"],
    }
    return {
        "observations": {
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "device_serial": protocol.device["serial"],
            "image_sha256": protocol.image["sha256"],
            "warmup_latency_ms": warmup,
            "latency_ms": measured,
            "correct": correct,
            "energy": energy,
            "notes": _text(observations.get("notes"), "notes", required=False),
        },
        "derived": {**derived, "latency_percentile_method": "nearest-rank"},
        "verdicts": [
            _judge(criterion, derived[criterion["metric"]]) for criterion in protocol.criteria
        ],
    }


def seal_receipt(protocol: Protocol, observations: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a hardware run's raw observations against its declared protocol.

    Parameters
    ----------
    protocol : Protocol
        The protocol declared before the run.
    observations : mapping
        ``started_at``, ``finished_at``, ``device_serial``, ``image_sha256``,
        ``warmup_latency_ms`` and ``latency_ms`` lists, ``correct`` and, when
        power was declared, ``energy`` (``source: instrument``,
        ``instrument_serial``, ``joules_per_inference`` per measured run);
        optional ``notes``.

    Returns
    -------
    dict
        The receipt: the full protocol, the admitted observations, derived
        accuracy, nearest-rank latency percentiles and energy, the verdict of
        every preregistered criterion, and ``sha256`` over all of it.

    Raises
    ------
    HardwareExperimentError
        An observation that does not belong to this protocol, or energy that
        was not measured by its declared, calibrated instrument.
    """
    body = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "protocol": protocol.to_public_dict(),
        "protocol_sha256": protocol.sha256,
        **_derive(protocol, _mapping(observations, "observations")),
    }
    return {**body, "sha256": _canonical_sha256(body)}


def verify_receipt(receipt: object) -> dict[str, Any]:
    """Recompute a sealed receipt from its protocol and raw observations.

    Parameters
    ----------
    receipt : mapping
        A document :func:`seal_receipt` produced.

    Returns
    -------
    dict
        The receipt, when every digest, derived figure and verdict recomputes
        to exactly what it states.

    Raises
    ------
    HardwareExperimentError
        Another schema, a changed protocol or observation, or a derived figure
        or verdict that does not follow from the observations.
    """
    data = _mapping(receipt, "receipt")
    if data.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise HardwareExperimentError("schema_version", "is not the receipt this build reads.")
    protocol = resolve_protocol(data.get("protocol"))
    if data.get("protocol_sha256") != protocol.sha256:
        raise HardwareExperimentError("protocol_sha256", "does not match the embedded protocol.")
    observations = dict(_mapping(data.get("observations"), "observations"))
    energy = observations.get("energy")
    if isinstance(energy, Mapping):
        observations["energy"] = {
            key: energy.get(key) for key in ("source", "instrument_serial", "joules_per_inference")
        }
    expected = seal_receipt(protocol, observations)
    if dict(data) != expected:
        raise HardwareExperimentError(
            "receipt", "its figures, verdicts or digest do not follow from its observations."
        )
    return expected


__all__ = [
    "CRITERIA",
    "IMAGE_KINDS",
    "PROTOCOL_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "HardwareExperimentError",
    "Protocol",
    "resolve_protocol",
    "seal_receipt",
    "verify_receipt",
]
