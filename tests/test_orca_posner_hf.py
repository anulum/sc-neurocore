# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fail-closed contracts for ORCA Posner HF calibration

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

orca = pytest.importorskip("orca_posner_hf")
acquire = pytest.importorskip("acquire_posner_external_data")


def test_auto_a_ref_calibrates_from_max_component() -> None:
    hf = [
        {"Axx": 10.0, "Ayy": -20.0, "Azz": 5.0, "Axy": 3.0, "Axz": -4.0, "Ayz": 1.0},
        {"Axx": 8.0, "Ayy": 15.0, "Azz": -12.0},
    ]
    assert orca.auto_a_ref(hf, target_max=0.5) == pytest.approx(40.0)
    assert orca.auto_a_ref(hf, target_max=0.25) == pytest.approx(80.0)


def test_auto_a_ref_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="empty HF tensor list"):
        orca.auto_a_ref([], target_max=0.5)


def test_auto_a_ref_rejects_missing_or_zero_components() -> None:
    with pytest.raises(ValueError, match="no hyperfine tensor components"):
        orca.auto_a_ref([{"atom_index": 1}], target_max=0.5)
    with pytest.raises(ValueError, match="components are zero"):
        orca.auto_a_ref([{"Axx": 0.0, "Ayy": 0.0, "Azz": 0.0}], target_max=0.5)


@pytest.mark.parametrize("target_max", [0.0, -0.1])
def test_auto_a_ref_rejects_non_positive_target_max(target_max: float) -> None:
    with pytest.raises(ValueError, match="target_max must be finite and > 0"):
        orca.auto_a_ref([{"Axx": 1.0}], target_max=target_max)


def test_convert_to_dimensionless_rejects_invalid_reference_scale() -> None:
    hf = [{"atom_index": 1, "Axx": 10.0, "Ayy": 5.0, "Azz": 1.0}]
    with pytest.raises(ValueError, match="a_ref_MHz must be finite and > 0"):
        orca.convert_to_dimensionless(hf, a_ref_MHz=0.0)


def test_run_full_pipeline_rejects_missing_explicit_orca_path(tmp_path) -> None:
    missing = tmp_path / "missing_orca.out"
    with pytest.raises(ValueError, match="ORCA output path does not exist"):
        orca.run_full_pipeline(missing)


def test_neutral_optimization_parser_refuses_normal_but_unconverged_endpoint(tmp_path) -> None:
    out = tmp_path / "neutral.out"
    out.write_text(
        """
GEOMETRY OPTIMIZATION CYCLE 117
FINAL SINGLE POINT ENERGY     -9954.015112995519
SCF CONVERGED AFTER 17 CYCLES
          ----------------------|Geometry convergence|-------------------------
          Item                value                   Tolerance       Converged
          ---------------------------------------------------------------------
          Energy change      -0.0000590901            0.0000050000      NO
          RMS gradient        0.0001032437            0.0001000000      NO
          MAX gradient        0.0010178376            0.0003000000      NO
          RMS step            0.0045728307            0.0020000000      NO
          MAX step            0.0470340637            0.0040000000      NO
          -------------------------------------------------------------------------
The optimization has not yet converged - more geometry cycles are needed
                             ****ORCA TERMINATED NORMALLY****
TOTAL RUN TIME: 22 days 16 hours 52 minutes 5 seconds 565 msec
""",
        encoding="utf-8",
    )

    status = acquire.parse_neutral_optimization_status(out, exit_status=0)

    assert status["accepted_neutral_geometry"] is False
    assert status["markers"]["orca_terminated_normally"] is True
    assert status["markers"]["the_optimization_has_converged"] is False
    assert status["markers"]["last_geometry_optimization_cycle"] == 117
    assert status["final_energy_Eh"] == pytest.approx(-9954.015112995519)
    assert status["final_geometry_convergence"]["all_items_converged"] is False


def test_neutral_optimization_parser_accepts_converged_normal_endpoint(tmp_path) -> None:
    out = tmp_path / "neutral.out"
    out.write_text(
        """
GEOMETRY OPTIMIZATION CYCLE 12
FINAL SINGLE POINT ENERGY     -123.5
SCF CONVERGED AFTER 8 CYCLES
          ----------------------|Geometry convergence|-------------------------
          Item                value                   Tolerance       Converged
          ---------------------------------------------------------------------
          Energy change      -0.0000000100            0.0000050000      YES
          RMS gradient        0.0000200000            0.0001000000      YES
          MAX gradient        0.0001000000            0.0003000000      YES
          RMS step            0.0002000000            0.0020000000      YES
          MAX step            0.0010000000            0.0040000000      YES
          -------------------------------------------------------------------------
THE OPTIMIZATION HAS CONVERGED
                             ****ORCA TERMINATED NORMALLY****
TOTAL RUN TIME: 1 days 0 hours 0 minutes 0 seconds 0 msec
""",
        encoding="utf-8",
    )

    status = acquire.parse_neutral_optimization_status(out, exit_status=0)

    assert status["accepted_neutral_geometry"] is True
    assert status["markers"]["the_optimization_has_converged"] is True
    assert status["final_geometry_convergence"]["all_items_converged"] is True
