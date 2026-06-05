# SPDX-License-Identifier: AGPL-3.0-or-later
"""DCLS Q8.8 reference and PyTorch cosimulation contract."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from cosim_dcls_q88_vs_pytorch import (
    dcls_q88_reference,
    run_deterministic_suite,
    tent_gate_q88,
)


def test_tent_gate_matches_delay_kernel_special_cases() -> None:
    assert tent_gate_q88(1, centre_q88=256, sigma_q88=512) == 256
    assert tent_gate_q88(0, centre_q88=256, sigma_q88=512) == 128
    assert tent_gate_q88(3, centre_q88=256, sigma_q88=512) == 0


def test_dcls_three_tap_accumulator_matches_hand_computation() -> None:
    result = dcls_q88_reference([1, 1, 1], [256, 128, -64], 256, 512)
    assert result["accumulator_q16_16"] == 57_344
    assert result["output_q88"] == 224
    assert result["active_tap_count"] == 3
    assert result["max_gate_q88"] == 256
    assert result["overflow"] is False


def test_dcls_invalid_sigma_fails_closed() -> None:
    with pytest.raises(ValueError, match="sigma must be positive"):
        dcls_q88_reference([1], [256], 0, 0)


def test_dcls_deterministic_suite_matches_pytorch_when_available() -> None:
    report = run_deterministic_suite(require_torch=False)
    assert report["cases_passed"] == report["case_count"]
    assert report["max_abs_accumulator_diff"] == 0
    if report["pytorch_available"]:
        assert all(item["pytorch"] is not None for item in report["comparisons"])
