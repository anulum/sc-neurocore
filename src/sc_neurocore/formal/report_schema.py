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

from .network_properties import (
    DenseLIFNetworkSpec,
    NetworkAntagonisticOutputExclusion,
    NetworkOutputTemporalSeparation,
    NetworkPopulationCoactivationCap,
    NetworkPopulationInactivityBound,
    NetworkPopulationSilenceAfterCoactivation,
    NetworkRateBound,
    NetworkRefractoryInvariant,
)

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
            raise FormalReportValidationError(
                "refractory.output_index must exist in network output"
            )

    antagonistic_payload = payload.get("antagonistic_exclusion")
    antagonistic: NetworkAntagonisticOutputExclusion | None = None
    if antagonistic_payload is not None:
        try:
            antagonistic = NetworkAntagonisticOutputExclusion(
                **_expect_mapping(antagonistic_payload, "antagonistic_exclusion")
            )
        except ValueError as exc:
            raise FormalReportValidationError(str(exc)) from exc
        if antagonistic.output_a >= network.output_width:
            raise FormalReportValidationError(
                "antagonistic_exclusion.output_a must exist in network output"
            )
        if antagonistic.output_b >= network.output_width:
            raise FormalReportValidationError(
                "antagonistic_exclusion.output_b must exist in network output"
            )

    temporal_payload = payload.get("temporal_separation")
    temporal: NetworkOutputTemporalSeparation | None = None
    if temporal_payload is not None:
        try:
            temporal = NetworkOutputTemporalSeparation(
                **_expect_mapping(temporal_payload, "temporal_separation")
            )
        except ValueError as exc:
            raise FormalReportValidationError(str(exc)) from exc
        if temporal.output_a >= network.output_width:
            raise FormalReportValidationError(
                "temporal_separation.output_a must exist in network output"
            )
        if temporal.output_b >= network.output_width:
            raise FormalReportValidationError(
                "temporal_separation.output_b must exist in network output"
            )

    population_payload = payload.get("population_coactivation")
    population: NetworkPopulationCoactivationCap | None = None
    if population_payload is not None:
        try:
            population = NetworkPopulationCoactivationCap(
                **_expect_mapping(population_payload, "population_coactivation")
            )
        except ValueError as exc:
            raise FormalReportValidationError(str(exc)) from exc
        if population.max_active_outputs > network.output_width:
            raise FormalReportValidationError(
                "population_coactivation.max_active_outputs must be <= network output_width"
            )

    population_silence_payload = payload.get("population_silence")
    population_silence: NetworkPopulationSilenceAfterCoactivation | None = None
    if population_silence_payload is not None:
        try:
            population_silence = NetworkPopulationSilenceAfterCoactivation(
                **_expect_mapping(population_silence_payload, "population_silence")
            )
        except ValueError as exc:
            raise FormalReportValidationError(str(exc)) from exc
        if population_silence.trigger_active_outputs > network.output_width:
            raise FormalReportValidationError(
                "population_silence.trigger_active_outputs must be <= network output_width"
            )

    population_inactivity_payload = payload.get("population_inactivity")
    population_inactivity: NetworkPopulationInactivityBound | None = None
    if population_inactivity_payload is not None:
        try:
            population_inactivity = NetworkPopulationInactivityBound(
                **_expect_mapping(population_inactivity_payload, "population_inactivity")
            )
        except ValueError as exc:
            raise FormalReportValidationError(f"population_inactivity.{exc}") from exc

    artifacts = _expect_mapping(payload.get("artifacts"), "artifacts")
    _validate_artifacts(
        artifacts,
        refractory=refractory,
        antagonistic=antagonistic,
        temporal=temporal,
        population=population,
        population_silence=population_silence,
        population_inactivity=population_inactivity,
        artifact_root=artifact_root,
    )
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
    if antagonistic is None:
        if payload.get("antagonistic_replay") is not None:
            raise FormalReportValidationError(
                "antagonistic_replay must be null when antagonistic_exclusion is null"
            )
    else:
        _validate_antagonistic_replay(
            payload.get("antagonistic_replay"),
            "antagonistic_replay",
            antagonistic=antagonistic,
        )
    if temporal is None:
        if payload.get("temporal_replay") is not None:
            raise FormalReportValidationError(
                "temporal_replay must be null when temporal_separation is null"
            )
    else:
        _validate_temporal_replay(
            payload.get("temporal_replay"),
            "temporal_replay",
            temporal=temporal,
        )
    if population is None:
        if payload.get("population_replay") is not None:
            raise FormalReportValidationError(
                "population_replay must be null when population_coactivation is null"
            )
    else:
        _validate_population_replay(
            payload.get("population_replay"),
            "population_replay",
            population=population,
            network=network,
        )
    if population_silence is None:
        if payload.get("population_silence_replay") is not None:
            raise FormalReportValidationError(
                "population_silence_replay must be null when population_silence is null"
            )
    else:
        _validate_population_silence_replay(
            payload.get("population_silence_replay"),
            "population_silence_replay",
            silence=population_silence,
            network=network,
        )
    if population_inactivity is None:
        if payload.get("population_inactivity_replay") is not None:
            raise FormalReportValidationError(
                "population_inactivity_replay must be null when population_inactivity is null"
            )
    else:
        _validate_population_inactivity_replay(
            payload.get("population_inactivity_replay"),
            "population_inactivity_replay",
            inactivity=population_inactivity,
        )

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
    antagonistic: NetworkAntagonisticOutputExclusion | None,
    temporal: NetworkOutputTemporalSeparation | None,
    population: NetworkPopulationCoactivationCap | None,
    population_silence: NetworkPopulationSilenceAfterCoactivation | None,
    population_inactivity: NetworkPopulationInactivityBound | None,
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
    antagonistic_sva = artifacts.get("antagonistic_sva")
    if antagonistic is None:
        if antagonistic_sva is not None:
            raise FormalReportValidationError(
                "artifacts.antagonistic_sva must be null when antagonistic_exclusion is null"
            )
    else:
        _expect_artifact_path(artifacts, "antagonistic_sva", artifact_root=artifact_root)
    temporal_sva = artifacts.get("temporal_sva")
    if temporal is None:
        if temporal_sva is not None:
            raise FormalReportValidationError(
                "artifacts.temporal_sva must be null when temporal_separation is null"
            )
    else:
        _expect_artifact_path(artifacts, "temporal_sva", artifact_root=artifact_root)
    population_sva = artifacts.get("population_sva")
    if population is None:
        if population_sva is not None:
            raise FormalReportValidationError(
                "artifacts.population_sva must be null when population_coactivation is null"
            )
    else:
        _expect_artifact_path(artifacts, "population_sva", artifact_root=artifact_root)
    population_silence_sva = artifacts.get("population_silence_sva")
    if population_silence is None:
        if population_silence_sva is not None:
            raise FormalReportValidationError(
                "artifacts.population_silence_sva must be null when population_silence is null"
            )
    else:
        _expect_artifact_path(artifacts, "population_silence_sva", artifact_root=artifact_root)
    population_inactivity_sva = artifacts.get("population_inactivity_sva")
    if population_inactivity is None:
        if population_inactivity_sva is not None:
            raise FormalReportValidationError(
                "artifacts.population_inactivity_sva must be null when population_inactivity is null"
            )
    else:
        _expect_artifact_path(
            artifacts,
            "population_inactivity_sva",
            artifact_root=artifact_root,
        )


def _validate_rate_replay(value: Any, field: str) -> None:
    if value is None:
        return
    replay = _expect_mapping(value, field)
    _expect_bool(replay.get("violated"), f"{field}.violated")
    _expect_optional_non_negative_int(
        replay.get("first_violation_cycle"), f"{field}.first_violation_cycle"
    )
    _expect_optional_non_negative_int(
        replay.get("window_start_cycle"), f"{field}.window_start_cycle"
    )
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


def _validate_antagonistic_replay(
    value: Any,
    field: str,
    *,
    antagonistic: NetworkAntagonisticOutputExclusion,
) -> None:
    if value is None:
        return
    replay = _expect_mapping(value, field)
    _expect_bool(replay.get("violated"), f"{field}.violated")
    _expect_optional_non_negative_int(
        replay.get("first_violation_cycle"), f"{field}.first_violation_cycle"
    )
    if (
        _expect_non_negative_int(replay.get("output_a"), f"{field}.output_a")
        != antagonistic.output_a
    ):
        raise FormalReportValidationError(f"{field}.output_a must match antagonistic_exclusion")
    if (
        _expect_non_negative_int(replay.get("output_b"), f"{field}.output_b")
        != antagonistic.output_b
    ):
        raise FormalReportValidationError(f"{field}.output_b must match antagonistic_exclusion")
    _expect_non_negative_int(replay.get("cycles_checked"), f"{field}.cycles_checked")


def _validate_temporal_replay(
    value: Any,
    field: str,
    *,
    temporal: NetworkOutputTemporalSeparation,
) -> None:
    if value is None:
        return
    replay = _expect_mapping(value, field)
    _expect_bool(replay.get("violated"), f"{field}.violated")
    _expect_optional_non_negative_int(
        replay.get("first_violation_cycle"), f"{field}.first_violation_cycle"
    )
    trigger_output = _expect_optional_non_negative_int(
        replay.get("trigger_output"), f"{field}.trigger_output"
    )
    violating_output = _expect_optional_non_negative_int(
        replay.get("violating_output"), f"{field}.violating_output"
    )
    temporal_outputs = {temporal.output_a, temporal.output_b}
    if trigger_output is not None and trigger_output not in temporal_outputs:
        raise FormalReportValidationError(
            f"{field}.trigger_output must match temporal_separation outputs"
        )
    if violating_output is not None and violating_output not in temporal_outputs:
        raise FormalReportValidationError(
            f"{field}.violating_output must match temporal_separation outputs"
        )
    if (
        trigger_output is not None
        and violating_output is not None
        and trigger_output == violating_output
    ):
        raise FormalReportValidationError(
            f"{field}.violating_output must differ from trigger_output"
        )
    _expect_non_negative_int(
        replay.get("remaining_separation_cycles"), f"{field}.remaining_separation_cycles"
    )
    _expect_non_negative_int(replay.get("cycles_checked"), f"{field}.cycles_checked")


def _validate_population_replay(
    value: Any,
    field: str,
    *,
    population: NetworkPopulationCoactivationCap,
    network: DenseLIFNetworkSpec,
) -> None:
    if value is None:
        return
    replay = _expect_mapping(value, field)
    violated = _expect_bool(replay.get("violated"), f"{field}.violated")
    first_violation_cycle = _expect_optional_non_negative_int(
        replay.get("first_violation_cycle"), f"{field}.first_violation_cycle"
    )
    observed = _expect_non_negative_int(
        replay.get("observed_active_outputs"), f"{field}.observed_active_outputs"
    )
    if observed > network.output_width:
        raise FormalReportValidationError(
            f"{field}.observed_active_outputs must be <= network output_width"
        )
    if (
        _expect_non_negative_int(replay.get("max_active_outputs"), f"{field}.max_active_outputs")
        != population.max_active_outputs
    ):
        raise FormalReportValidationError(
            f"{field}.max_active_outputs must match population_coactivation"
        )
    if violated and observed <= population.max_active_outputs:
        raise FormalReportValidationError(
            f"{field}.observed_active_outputs must exceed max_active_outputs when violated"
        )
    if violated and first_violation_cycle is None:
        raise FormalReportValidationError(
            f"{field}.first_violation_cycle must be present when violated"
        )
    if not violated and first_violation_cycle is not None:
        raise FormalReportValidationError(
            f"{field}.first_violation_cycle must be null when not violated"
        )
    if not violated and observed > population.max_active_outputs:
        raise FormalReportValidationError(
            f"{field}.observed_active_outputs must be <= max_active_outputs when not violated"
        )
    _expect_non_negative_int(replay.get("cycles_checked"), f"{field}.cycles_checked")


def _validate_population_silence_replay(
    value: Any,
    field: str,
    *,
    silence: NetworkPopulationSilenceAfterCoactivation,
    network: DenseLIFNetworkSpec,
) -> None:
    if value is None:
        return
    replay = _expect_mapping(value, field)
    violated = _expect_bool(replay.get("violated"), f"{field}.violated")
    first_violation_cycle = _expect_optional_non_negative_int(
        replay.get("first_violation_cycle"), f"{field}.first_violation_cycle"
    )
    trigger_cycle = _expect_optional_non_negative_int(
        replay.get("trigger_cycle"), f"{field}.trigger_cycle"
    )
    observed = _expect_non_negative_int(
        replay.get("observed_active_outputs"), f"{field}.observed_active_outputs"
    )
    if observed > network.output_width:
        raise FormalReportValidationError(
            f"{field}.observed_active_outputs must be <= network output_width"
        )
    remaining_silence_cycles = _expect_non_negative_int(
        replay.get("remaining_silence_cycles"), f"{field}.remaining_silence_cycles"
    )
    if (
        _expect_non_negative_int(
            replay.get("trigger_active_outputs"), f"{field}.trigger_active_outputs"
        )
        != silence.trigger_active_outputs
    ):
        raise FormalReportValidationError(
            f"{field}.trigger_active_outputs must match population_silence"
        )
    if _expect_non_negative_int(replay.get("silence_cycles"), f"{field}.silence_cycles") != (
        silence.silence_cycles
    ):
        raise FormalReportValidationError(f"{field}.silence_cycles must match population_silence")
    if remaining_silence_cycles > silence.silence_cycles:
        raise FormalReportValidationError(
            f"{field}.remaining_silence_cycles must be <= population_silence.silence_cycles"
        )
    if violated and first_violation_cycle is None:
        raise FormalReportValidationError(
            f"{field}.first_violation_cycle must be present when violated"
        )
    if violated and observed == 0:
        raise FormalReportValidationError(
            f"{field}.observed_active_outputs must be positive when violated"
        )
    if not violated and first_violation_cycle is not None:
        raise FormalReportValidationError(
            f"{field}.first_violation_cycle must be null when not violated"
        )
    if not violated and observed != 0:
        raise FormalReportValidationError(
            f"{field}.observed_active_outputs must be zero when not violated"
        )
    cycles_checked = _expect_non_negative_int(
        replay.get("cycles_checked"), f"{field}.cycles_checked"
    )
    if first_violation_cycle is not None and first_violation_cycle >= cycles_checked:
        raise FormalReportValidationError(
            f"{field}.first_violation_cycle must be less than cycles_checked"
        )
    if trigger_cycle is not None and trigger_cycle >= cycles_checked:
        raise FormalReportValidationError(f"{field}.trigger_cycle must be less than cycles_checked")
    if violated and trigger_cycle is None:
        raise FormalReportValidationError(f"{field}.trigger_cycle must be present when violated")
    if (
        violated
        and trigger_cycle is not None
        and first_violation_cycle is not None
        and trigger_cycle >= first_violation_cycle
    ):
        raise FormalReportValidationError(
            f"{field}.trigger_cycle must precede first_violation_cycle"
        )
    if (
        violated
        and trigger_cycle is not None
        and first_violation_cycle is not None
        and first_violation_cycle - trigger_cycle > silence.silence_cycles
    ):
        raise FormalReportValidationError(
            f"{field}.first_violation_cycle must be within population_silence.silence_cycles"
        )
    if trigger_cycle is None:
        expected_remaining_silence_cycles = 0
    elif violated and first_violation_cycle is not None:
        elapsed_silence_cycles = first_violation_cycle - trigger_cycle - 1
        expected_remaining_silence_cycles = silence.silence_cycles - elapsed_silence_cycles
    else:
        elapsed_silence_cycles = cycles_checked - trigger_cycle - 1
        expected_remaining_silence_cycles = max(0, silence.silence_cycles - elapsed_silence_cycles)
    if remaining_silence_cycles != expected_remaining_silence_cycles:
        raise FormalReportValidationError(
            f"{field}.remaining_silence_cycles must match population_silence replay timing"
        )


def _validate_population_inactivity_replay(
    value: Any,
    field: str,
    *,
    inactivity: NetworkPopulationInactivityBound,
) -> None:
    if value is None:
        return
    replay = _expect_mapping(value, field)
    violated = _expect_bool(replay.get("violated"), f"{field}.violated")
    first_violation_cycle = _expect_optional_non_negative_int(
        replay.get("first_violation_cycle"), f"{field}.first_violation_cycle"
    )
    observed_silent_cycles = _expect_non_negative_int(
        replay.get("observed_silent_cycles"), f"{field}.observed_silent_cycles"
    )
    if (
        _expect_non_negative_int(replay.get("max_silent_cycles"), f"{field}.max_silent_cycles")
        != inactivity.max_silent_cycles
    ):
        raise FormalReportValidationError(
            f"{field}.max_silent_cycles must match population_inactivity"
        )
    if violated and first_violation_cycle is None:
        raise FormalReportValidationError(
            f"{field}.first_violation_cycle must be present when violated"
        )
    if not violated and first_violation_cycle is not None:
        raise FormalReportValidationError(
            f"{field}.first_violation_cycle must be null when not violated"
        )
    if violated and observed_silent_cycles <= inactivity.max_silent_cycles:
        raise FormalReportValidationError(
            f"{field}.observed_silent_cycles must exceed max_silent_cycles when violated"
        )
    if violated and observed_silent_cycles != inactivity.max_silent_cycles + 1:
        raise FormalReportValidationError(
            f"{field}.observed_silent_cycles must match first violation timing"
        )
    if not violated and observed_silent_cycles > inactivity.max_silent_cycles:
        raise FormalReportValidationError(
            f"{field}.observed_silent_cycles must be <= max_silent_cycles when not violated"
        )
    cycles_checked = _expect_non_negative_int(
        replay.get("cycles_checked"), f"{field}.cycles_checked"
    )
    if first_violation_cycle is not None and first_violation_cycle >= cycles_checked:
        raise FormalReportValidationError(
            f"{field}.first_violation_cycle must be less than cycles_checked"
        )


def _expect_artifact_path(
    artifacts: dict[str, Any],
    key: str,
    *,
    artifact_root: str | Path | None,
) -> None:
    value = _expect_str(artifacts.get(key), f"artifacts.{key}")
    if artifact_root is None:
        return
    path = Path(value).expanduser()
    if path.is_symlink():
        raise FormalReportValidationError(f"artifacts.{key} must not be a symlink: {path}")
    if not path.exists():
        raise FormalReportValidationError(f"artifacts.{key} does not exist: {path}")
    if not path.is_file():
        raise FormalReportValidationError(f"artifacts.{key} must be a regular file: {path}")
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
