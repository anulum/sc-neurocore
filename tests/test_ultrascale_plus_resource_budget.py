# SPDX-License-Identifier: AGPL-3.0-or-later
"""UltraScale+ SKU budget and resource-estimate contract."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from gen_vivado_project import SUPPORTED_SKUS, sku_baseline


def test_zu3eg_and_zu9eg_budget_baselines_are_nonzero_and_dsp48e2() -> None:
    baseline = sku_baseline()
    assert baseline["zu3eg"]["part"] == "xczu3eg-sbva484-1-e"
    assert baseline["zu3eg"]["dsp_budget"] == 360
    assert baseline["zu3eg"]["bram_36k_budget"] == 216
    assert baseline["zu3eg"]["dsp_primitive"] == "DSP48E2"
    assert baseline["zu9eg"]["part"] == "xczu9eg-ffvb1156-2-e"
    assert baseline["zu9eg"]["dsp_budget"] == 2520
    assert baseline["zu9eg"]["bram_36k_budget"] == 912
    assert baseline["zu9eg"]["dsp_primitive"] == "DSP48E2"


def test_dense_shd_scale_estimate_fits_zu3eg_budget_boundary() -> None:
    sku = SUPPORTED_SKUS["zu3eg"]
    n_inputs = 64
    n_outputs = 32
    dense_macs = n_inputs * n_outputs
    estimated_bram_36k = ((dense_macs * 16) + 36_863) // 36_864
    assert dense_macs > sku.dsp_budget
    assert estimated_bram_36k <= sku.bram_36k_budget


def test_small_control_dense_estimate_fits_zu3eg_budget() -> None:
    sku = SUPPORTED_SKUS["zu3eg"]
    n_inputs = 8
    n_outputs = 8
    dense_macs = n_inputs * n_outputs
    estimated_bram_36k = ((dense_macs * 16) + 36_863) // 36_864
    assert dense_macs <= sku.dsp_budget
    assert estimated_bram_36k <= sku.bram_36k_budget
