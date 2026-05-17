# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Formal network verification report schema

from __future__ import annotations

from pathlib import Path
from typing import Any

from .network_properties import DenseLIFNetworkSpec, NetworkRateBound, NetworkRefractoryInvariant

FORMAL_NETWORK_REPORT_SCHEMA_VERSION = "sc-neurocore.formal-network-rate-bound.v0.1"

_SYMBIYOSYS_STATUSES = {"not_requested", "tool_unavailable", "passed", "failed"}


class FormalReportValidationError(ValueError):
    """Raised when a formal network verification report violates its schema."""


def validate_formal_network_report(
    payload: dict[str, Any],
    *,
    artifact_root: str | Path | None = None,
) -> None:
    """Validate a formal network verification report without external dependencies."""
    _expect_mapping(payload, "report")
    _expect_equal(
        payload.get("schema_version"),
        FORMAL_NETWORK_REPORT_SCHEMA_VERSION,
        "schema_version",
    )

    try:
        network = DenseLIFNetworkSpec(**_expect_mapping(payload.get("network"), "network"))
        rate_bound = NetworkRateBound(**_expect_mapping(payload.get("rate_bound"), "rate_bound"))
    except ValueError as exc:
        raise FormalReportValidationError(str(exc)) from exc
    if rate_bound.output_index >= network.output_width:
        raise FormalReportValidationError("rate_bound.output_index must exist in network output")

    refractory_payload = payload.get("refractory")
    refractory: NetworkRefractoryInvariant | None = None
    if refractory_payload is not None:
        try:
            refractory = NetworkRefractoryInvariant(
                **_expect_mapping(refractory_payload, "refractory")
            )
        except ValueError as exc:
            raise FormalReportValidationError(str(exc)) from exc
        if refractory.output_index >= network.output_width:
            raise FormalReportValidationError("refractory.output_index must exist in network output")

    artifacts = _expect_mapping(payload.get("artifacts"), "artifacts")
    _validate_artifacts(artifacts, refractory=refractory, artifact_root=artifact_root)
    _validate_rate_replay(payload.get("rate_replay"), "rate_replay")
    if payload.get("replay") != payload.get("rate_replay"):
        raise FormalReportValidationError("replay must match rate_replay")
    if refractory is None:
        if payload.get("refractory_replay") is not None:
            raise FormalReportValidationError(
                "refractory_replay must be null when refractory is null"
            )
    else:
        _validate_refractory_replay(payload.get("refractory_replay"), "refractory_replay")

    symbiyosys = _expect_mapping(payload.get("symbiyosys"), "symbiyosys")
    status = _expect_str(symbiyosys.get("status"), "symbiyosys.status")
    if status not in _SYMBIYOSYS_STATUSES:
        raise FormalReportValidationError("symbiyosys.status has unsupported value")
    if not isinstance(symbiyosys.get("requested"), bool):
        raise FormalReportValidationError("symbiyosys.requested must be a boolean")
    if symbiyosys.get("returncode") is not None and not isinstance(
        symbiyosys.get("returncode"), int
    ):
        raise FormalReportValidationError("symbiyosys.returncode must be int or null")
    _expect_string(symbiyosys.get("stdout"), "symbiyosys.stdout")
    _expect_string(symbiyosys.get("stderr"), "symbiyosys.stderr")
    if symbiyosys.get("sby") != artifacts["sby"]:
        raise FormalReportValidationError("symbiyosys.sby must match artifacts.sby")


def _validate_artifacts(
    artifacts: dict[str, Any],
    *,
    refractory: NetworkRefractoryInvariant | None,
    artifact_root: str | Path | None,
) -> None:
    required = ("rtl", "sva", "rate_sva", "formal_bundle", "sby", "report")
    for key in required:
        _expect_artifact_path(artifacts, key, artifact_root=artifact_root)
    if artifacts["sva"] != artifacts["rate_sva"]:
        raise FormalReportValidationError("artifacts.sva must match artifacts.rate_sva")
    refractory_sva = artifacts.get("refractory_sva")
    if refractory is None:
        if refractory_sva is not None:
            raise FormalReportValidationError(
                "artifacts.refractory_sva must be null when refractory is null"
            )
    else:
        _expect_artifact_path(artifacts, "refractory_sva", artifact_root=artifact_root)


def _validate_rate_replay(value: Any, field: str) -> None:
    if value is None:
        return
    replay = _expect_mapping(value, field)
    _expect_bool(replay.get("violated"), f"{field}.violated")
    _expect_optional_non_negative_int(
        replay.get("first_violation_cycle"), f"{field}.first_violation_cycle"
    )
    _expect_optional_non_negative_int(replay.get("window_start_cycle"), f"{field}.window_start_cycle")
    _expect_non_negative_int(replay.get("observed_spikes"), f"{field}.observed_spikes")
    _expect_non_negative_int(replay.get("cycles_checked"), f"{field}.cycles_checked")


def _validate_refractory_replay(value: Any, field: str) -> None:
    if value is None:
        return
    replay = _expect_mapping(value, field)
    _expect_bool(replay.get("violated"), f"{field}.violated")
    _expect_optional_non_negative_int(
        replay.get("first_violation_cycle"), f"{field}.first_violation_cycle"
    )
    _expect_optional_non_negative_int(replay.get("trigger_cycle"), f"{field}.trigger_cycle")
    _expect_non_negative_int(
        replay.get("remaining_refractory_cycles"), f"{field}.remaining_refractory_cycles"
    )
    _expect_non_negative_int(replay.get("cycles_checked"), f"{field}.cycles_checked")


def _expect_artifact_path(
    artifacts: dict[str, Any],
    key: str,
    *,
    artifact_root: str | Path | None,
) -> None:
    value = _expect_str(artifacts.get(key), f"artifacts.{key}")
    if artifact_root is None:
        return
    path = Path(value)
    if not path.exists():
        raise FormalReportValidationError(f"artifacts.{key} does not exist: {path}")
    root = Path(artifact_root).resolve()
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise FormalReportValidationError(f"artifacts.{key} is outside artifact_root") from exc


def _expect_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FormalReportValidationError(f"{field} must be an object")
    return value


def _expect_equal(value: Any, expected: Any, field: str) -> None:
    if value != expected:
        raise FormalReportValidationError(f"{field} must be {expected!r}")


def _expect_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise FormalReportValidationError(f"{field} must be a non-empty string")
    return value


def _expect_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise FormalReportValidationError(f"{field} must be a string")
    return value


def _expect_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise FormalReportValidationError(f"{field} must be a boolean")
    return value


def _expect_non_negative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FormalReportValidationError(f"{field} must be a non-negative integer")
    return value


def _expect_optional_non_negative_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _expect_non_negative_int(value, field)
