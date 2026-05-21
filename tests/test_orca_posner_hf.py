# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fail-closed contracts for ORCA Posner HF calibration

from __future__ import annotations

import pytest

orca = pytest.importorskip("orca_posner_hf")


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
