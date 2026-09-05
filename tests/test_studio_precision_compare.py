# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio float64 versus bit-true fixed-point comparison tests

"""The precision comparison runs genuine fixed-point arithmetic and refuses clamps.

The bit-true run is the generated C kernel compiled with the host compiler,
so these tests skip without ``gcc``; the rejection and contract tests run
everywhere.
"""

from __future__ import annotations

import shutil
from typing import Any

import numpy as np
import pytest
from starlette.testclient import TestClient

from sc_neurocore.compiler.intelligence.bit_true_kernel import (
    BIT_TRUE_ARITHMETIC_SCHEMA_VERSION,
    c_word_type,
    kernel_arithmetic_contract,
)
from sc_neurocore.neurons.equation_builder import from_equations
from sc_neurocore.studio import bit_true_execution
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.bit_true_execution import (
    NativeExecutionError,
    NativeToolUnavailable,
    harness_main,
    require_native_tools,
    run_bittrue_kernel,
)
from sc_neurocore.studio.model_run_contract import ModelInputError
from sc_neurocore.studio.precision_compare import (
    PRECISION_COMPARE_SCHEMA_VERSION,
    precision_compare,
    resolve_word_format,
)

HAS_GCC = shutil.which("gcc") is not None
requires_gcc = pytest.mark.skipif(not HAS_GCC, reason="gcc is required for the bit-true run")

LIF: dict[str, Any] = {
    "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
    "threshold": "v > -50",
    "reset": "v = -65",
    "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
    "init": {"v": -65.0},
    "dt": 0.1,
    "duration": 50.0,
    "current": 30.0,
}


@pytest.mark.parametrize(
    "override,field",
    [
        ({"dt": 0.0}, "dt"),
        ({"dt": -0.1}, "dt"),
        ({"dt": float("nan")}, "dt"),
        ({"duration": float("inf")}, "duration"),
        ({"duration": 0.0}, "duration"),
        ({"duration": -1.0}, "duration"),
        ({"duration": 0.01}, "duration"),
        ({"max_steps": 0}, "max_steps"),
        ({"max_steps": True}, "max_steps"),
        ({"max_steps": 2}, "duration"),
    ],
)
def test_precision_refuses_invalid_or_shortened_experiment(
    override: dict[str, Any], field: str
) -> None:
    """A complete metric must never describe only a prefix of the requested run."""
    with pytest.raises(ModelInputError) as error:
        precision_compare(**{**LIF, **override})
    assert error.value.field == field


@requires_gcc
def test_precision_runs_exact_step_limit_without_shortening() -> None:
    result = precision_compare(**{**LIF, "duration": 0.2, "max_steps": 2})
    for key in ("float_result", "fixed_result", "parameter_quantisation_result"):
        assert result[key]["n_steps"] == 2
        assert len(result[key]["raw"]["states"]["v"]) == 2


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")


def test_kernel_arithmetic_contract_states_the_operation_semantics() -> None:
    """The arithmetic statement comes from the generator's own configuration checks."""
    contract = kernel_arithmetic_contract()
    assert contract["schema_version"] == BIT_TRUE_ARITHMETIC_SCHEMA_VERSION
    assert contract["q_format"] == "Q8.8"
    assert contract["data_width"] == 16 and contract["fraction"] == 8
    assert contract["resolution"] == pytest.approx(1 / 256)
    assert contract["max_value"] == pytest.approx(127.99609375)
    assert "wrapped to 32 bits" in str(contract["multiply"])
    assert "saturation" in str(contract["state_update"])
    assert "candidate" in str(contract["sequencing"])
    assert str(contract["randomness"]).startswith("none")
    nearest_wrap = kernel_arithmetic_contract(overflow="wrap", rounding="nearest", method="map")
    assert "half a unit" in str(nearest_wrap["multiply"])
    assert "wrap of the sum" in str(nearest_wrap["state_update"])
    assert str(nearest_wrap["state_update"]).startswith("next = commit(f(state))")
    assert kernel_arithmetic_contract(data_width=32, fraction=16)["q_format"] == "Q16.16"
    with pytest.raises(ValueError, match="rounding"):
        kernel_arithmetic_contract(rounding="bankers")
    with pytest.raises(ValueError, match="method"):
        kernel_arithmetic_contract(method="rk4")
    with pytest.raises(ValueError, match="2\\*data_width <= 64"):
        kernel_arithmetic_contract(data_width=64, fraction=32)
    assert c_word_type(16) == "int16_t" and c_word_type(24) == "int32_t"


def test_resolve_word_format_rejects_unsupported_widths() -> None:
    assert resolve_word_format("Q8.8") == (16, 8)
    assert resolve_word_format("Q16.16") == (32, 16)
    assert resolve_word_format("Q4.12") == (16, 12)
    with pytest.raises(ModelInputError, match="not a Q"):
        resolve_word_format("float")
    with pytest.raises(ModelInputError, match="8 to 32"):
        resolve_word_format("Q32.32")
    with pytest.raises(ModelInputError) as info:
        resolve_word_format("Q0.16")
    assert info.value.field == "q_format"


@pytest.mark.parametrize(
    "override, field",
    [
        ({"params": {"E_L": -65.0, "tau_m": 200.0, "C": 1.0}}, "params.tau_m"),
        ({"init": {"v": -200.0}}, "init.v"),
        ({"current": 150.0}, "current"),
        ({"threshold": "v > 300"}, "threshold"),
        ({"reset": "v = -300"}, "reset"),
        ({"equations": ["dv/dt = -(v - E_L) / tau_m + I / 0.001"]}, "equations"),
        ({"dt": 0.001}, "dt"),
        ({"q_format": "Q40.8"}, "q_format"),
        ({"overflow": "trap"}, "overflow"),
        ({"rounding": "bankers"}, "rounding"),
        ({"equations": ["dv/dt = -(v - E_L) / tau_m + I / C + xi"]}, "equations"),
    ],
)
def test_precision_compare_rejects_values_the_word_cannot_hold(
    override: dict[str, Any], field: str
) -> None:
    """An unrepresentable value names its field; nothing is clamped or estimated."""
    request = {**LIF, **override}
    with pytest.raises(ModelInputError) as info:
        precision_compare(**request)
    assert info.value.field == field


def test_precision_compare_reports_tool_absence_instead_of_estimating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bit_true_execution, "resolve_native_tool", lambda name: None)
    with pytest.raises(NativeToolUnavailable) as info:
        precision_compare(**LIF)
    assert info.value.to_public_detail() == {
        "error": "native_tool_unavailable",
        "tools": ["gcc"],
        "reason": "Bit-true fixed-point execution needs native tools unavailable on this host: gcc.",
    }
    with pytest.raises(NativeToolUnavailable):
        require_native_tools(["gcc"], purpose="test")


def test_run_bittrue_kernel_rejects_words_outside_the_format() -> None:
    neuron = from_equations(*LIF["equations"], params=LIF["params"], init=LIF["init"], dt=0.1)
    with pytest.raises(ValueError, match="outside the 16-bit range"):
        run_bittrue_kernel(
            neuron,
            data_width=16,
            fraction=8,
            overflow="saturate",
            rounding="truncate",
            drive_words=[40_000],
        )
    with pytest.raises(ValueError, match="non-empty"):
        run_bittrue_kernel(
            neuron,
            data_width=16,
            fraction=8,
            overflow="saturate",
            rounding="truncate",
            drive_words=[],
        )
    harness = harness_main(neuron, "k", 16)
    assert "k_step(&st, (int16_t)word)" in harness
    assert "row[1] = (int64_t)st.v_out;" in harness


def test_native_command_failures_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    monkeypatch.setattr(
        "sc_neurocore.studio.bit_true_execution.subprocess.run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            args=["gcc"], returncode=1, stdout="", stderr="x"
        ),
    )
    with pytest.raises(NativeExecutionError, match="exited 1: x"):
        bit_true_execution.run_native_command(["/usr/bin/gcc"], timeout_seconds=1)
    assert bit_true_execution.native_tool_version("/usr/bin/gcc", "gcc") == (
        "available-version-unreported"
    )


@requires_gcc
def test_bit_true_run_compares_every_state_and_event_against_float64() -> None:
    """The fixed run is the kernel's decoded words; all variables and spikes are compared."""
    result = precision_compare(
        ["dv/dt = (-(v - E_L) - w + I) / tau_m", "dw/dt = (a * (v - E_L) - w) / tau_w"],
        "v > -50",
        "v = -65; w = w + b",
        {"E_L": -65.0, "tau_m": 10.0, "tau_w": 30.0, "a": 0.5, "b": 2.0},
        {"v": -65.0, "w": 0.0},
        0.1,
        60.0,
        20.0,
    )
    assert result["schema_version"] == PRECISION_COMPARE_SCHEMA_VERSION
    assert result["contract"]["kind"] == "precision-compare"
    assert result["fixed_result"]["backend"] == "bit-true-kernel"
    assert result["fixed_result"]["state_layout"]["source"] == "equations"
    words = result["fixed_result"]["words"]
    assert set(words["state"]) == {"v", "w"}
    n_steps = result["float_result"]["n_steps"]
    assert len(words["state"]["v"]) == n_steps == len(words["drive"])
    decoded = np.asarray(words["state"]["v"], dtype=np.float64) / 256.0
    assert decoded.tolist() == result["fixed_result"]["raw"]["states"]["v"]
    comparison = result["comparison"]
    assert set(comparison["bit_true"]["variables"]) == {"v", "w"}
    assert set(comparison["parameter_quantisation"]["variables"]) == {"v", "w"}
    for block in ("bit_true", "parameter_quantisation"):
        for metrics in comparison[block]["variables"].values():
            assert len(metrics["trace"]) == n_steps
            assert len(metrics["display"]) == len(result["float_result"]["time"])
            assert metrics["max_abs_error"] >= metrics["final_abs_error"] >= 0.0
    events = comparison["bit_true"]["events"]
    assert events["reference_count"] == result["float_result"]["spike_count"]
    assert events["candidate_count"] == result["fixed_result"]["spike_count"]
    assert result["error"]["kind"] == "bit-true-vs-float64"
    assert result["error"]["variable"] == "v"
    assert result["error"]["display"] == comparison["bit_true"]["variables"]["v"]["display"]
    assert result["arithmetic"]["q_format"] == "Q8.8"
    assert result["arithmetic"]["kernel_sha256"] != result["arithmetic"]["harness_sha256"]
    assert result["encoding"]["dt"]["word"] == 26
    assert result["encoding"]["params"]["tau_w"]["quantised"] == 30.0
    assert result["quantized_params"]["a"] == 0.5
    assert result["parameter_quantisation_result"]["dt"] == 0.1
    assert result["encoding"]["dt"]["quantised"] == pytest.approx(26 / 256)


@requires_gcc
def test_bit_true_saturation_is_distinguishable_from_parameter_rounding() -> None:
    """An integrator driven past the word range saturates in the kernel only."""
    saturating = precision_compare(["dv/dt = I"], None, None, {}, {"v": 0.0}, 0.1, 50.0, 100.0)
    per_variable = saturating["comparison"]["bit_true"]["saturation"]["per_variable"]["v"]
    assert per_variable["steps_at_max"] > 0
    assert saturating["fixed_result"]["final_state"]["v"] == pytest.approx(127.99609375)
    bit_true_error = saturating["comparison"]["bit_true"]["variables"]["v"]["max_abs_error"]
    parameter_error = saturating["comparison"]["parameter_quantisation"]["variables"]["v"][
        "max_abs_error"
    ]
    assert bit_true_error > 1000.0 > parameter_error

    wrapping = precision_compare(
        ["dv/dt = I"], None, None, {}, {"v": 0.0}, 0.1, 50.0, 100.0, overflow="wrap"
    )
    assert wrapping["arithmetic"]["overflow"] == "wrap"
    assert wrapping["fixed_result"]["final_state"]["v"] < 0.0
    assert wrapping["comparison"]["bit_true"]["saturation"]["per_variable"]["v"] == {
        "steps_at_max": 0,
        "steps_at_min": 0,
    }


@requires_gcc
def test_nearest_rounding_and_wider_word_reduce_the_error() -> None:
    truncate = precision_compare(**LIF)
    wide = precision_compare(**{**LIF, "q_format": "Q16.16"})
    assert (
        wide["comparison"]["bit_true"]["variables"]["v"]["max_abs_error"]
        < (truncate["comparison"]["bit_true"]["variables"]["v"]["max_abs_error"])
    )
    assert wide["arithmetic"]["q_format"] == "Q16.16"
    nearest = precision_compare(**{**LIF, "rounding": "nearest"})
    assert nearest["arithmetic"]["rounding"] == "nearest"
    assert nearest["arithmetic"]["kernel_sha256"] != truncate["arithmetic"]["kernel_sha256"]


@requires_gcc
def test_sine_protocol_feeds_the_same_drive_to_every_run() -> None:
    result = precision_compare(**{**LIF, "protocol": "sine", "frequency_hz": 40.0})
    drive = result["encoding"]["drive"]
    assert drive["protocol"] == "sine" and drive["frequency_hz"] == 40.0
    assert drive["max_abs_error"] <= 1 / 512
    assert (
        result["float_result"]["raw"]["drive"][:3]
        == result["parameter_quantisation_result"]["raw"]["drive"][:3]
    )
    quantised = np.asarray(result["fixed_result"]["raw"]["drive"])
    requested = np.asarray(result["float_result"]["raw"]["drive"])
    assert np.max(np.abs(quantised - requested)) <= 1 / 512


@requires_gcc
def test_precision_route_returns_contract_metadata_and_refuses_clamps(client: TestClient) -> None:
    response = client.post("/api/precision", json=LIF)
    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis_metadata"]["analysis_type"] == "precision"
    assert payload["analysis_metadata"]["contract"] == "precision-compare"
    assert payload["analysis_metadata"]["domain"] == "complete"
    assert payload["fixed_result"]["backend"] == "bit-true-kernel"

    rejected = client.post(
        "/api/precision", json={**LIF, "params": {**LIF["params"], "tau_m": 200.0}}
    )
    assert rejected.status_code == 422
    assert rejected.json()["detail"]["field"] == "params.tau_m"

    forbidden = client.post("/api/precision", json={**LIF, "unknown": 1})
    assert forbidden.status_code == 422

    cosim = client.post("/api/ir/cosim", json={**LIF, "q_format": "Q16.16"})
    assert cosim.status_code == 200
    assert cosim.json()["arithmetic"]["q_format"] == "Q16.16"


def test_precision_route_reports_missing_compiler_as_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bit_true_execution, "resolve_native_tool", lambda name: None)
    response = client.post("/api/precision", json=LIF)
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "native_tool_unavailable"
    assert response.json()["detail"]["tools"] == ["gcc"]
    cosim = client.post("/api/ir/cosim", json=LIF)
    assert cosim.status_code == 503
