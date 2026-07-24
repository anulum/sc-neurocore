# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (error_report) from former test_kuramoto_rtl.py

from __future__ import annotations

from tests.test_hdl_gen.kuramoto_rtl_support import *  # noqa: F403

def test_kuramoto_rtl_error_report_cli_writes_deterministic_gate(tmp_path: Path) -> None:
    output = tmp_path / "kuramoto_rtl_fixed_point_error.json"

    result = subprocess.run(
        [
            sys.executable,
            "tools/run_kuramoto_rtl_error_report.py",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["report_type"] == "kuramoto_rtl_fixed_point_error"
    assert payload["hardware_measurement_claimed"] is False
    assert payload["failed_cases"] == 0
    assert payload["passed_cases"] == payload["total_cases"]
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert {case["case_name"] for case in payload["cases"]} >= {
        "higher_coupling_quartet_short",
        "low_coupling_quartet_short",
        "no_coupling_single_oscillator",
        "wrap_boundary_pair",
        "near_antiphase_pair",
    }
    assert all(case["config"]["phase_regime"] for case in payload["cases"])

    higher_coupling = next(
        case for case in payload["cases"] if case["case_name"] == "higher_coupling_quartet_short"
    )
    assert higher_coupling["passed"] is True
    assert higher_coupling["rust_solver_parity"]["available"] is True
    assert (
        higher_coupling["rust_solver_parity"]["max_abs_phase_error_rad"]
        < higher_coupling["thresholds"]["rust_max_abs_phase_error_rad"]
    )
    assert (
        higher_coupling["rust_solver_parity"]["order_parameter_error"]
        < higher_coupling["thresholds"]["rust_order_parameter_error"]
    )
    assert higher_coupling["config"]["oscillator_count"] == 4
    assert higher_coupling["config"]["coupling"] == 1.0
    assert (
        higher_coupling["summary"]["max_abs_phase_error_rad"]
        < higher_coupling["thresholds"]["max_abs_phase_error_rad"]
    )
    assert (
        higher_coupling["summary"]["rms_phase_error_rad"]
        < higher_coupling["thresholds"]["rms_phase_error_rad"]
    )


def test_committed_kuramoto_rtl_error_report_matches_reference() -> None:
    report_path = Path("benchmarks/results/kuramoto_rtl_fixed_point_error.json")

    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["failed_cases"] == 0
    assert payload["total_cases"] >= 5
    for case in payload["cases"]:
        config = case["config"]
        assert config["phase_regime"] in {
            "nominal_higher_coupling",
            "nominal_low_coupling",
            "single_oscillator_no_coupling",
            "phase_modulus_wrap_boundary",
            "near_antiphase_circular_error",
        }
        emitter = KuramotoEmitter(
            n_oscillators=config["oscillator_count"],
            omegas=config["omegas"],
            initial_phases=config["initial_phases"],
            coupling=config["coupling"],
            dt=config["dt"],
            data_width=config["data_width"],
            fraction=config["fraction"],
            lut_size=config["lut_size"],
        )
        assert case["summary"] == emitter.fixed_point_error_summary(steps=config["steps"])
        assert case["rust_solver_parity"]["available"] is True
        assert (
            case["rust_solver_parity"]["max_abs_phase_error_rad"]
            < case["thresholds"]["rust_max_abs_phase_error_rad"]
        )
        assert (
            case["rust_solver_parity"]["order_parameter_error"]
            < case["thresholds"]["rust_order_parameter_error"]
        )
        assert case["passed"] is True


