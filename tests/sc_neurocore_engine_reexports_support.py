# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared engine re-export contract fixtures

"""Shared symbol inventories and availability probes for engine re-export tests."""

from __future__ import annotations

import pytest

_engine = pytest.importorskip("sc_neurocore_engine")

QA_SYMBOLS: tuple[str, ...] = (
    "py_qa_batch_ising_energy",
    "py_qa_gauge_transform",
    "py_qa_generate_gauges",
    "py_qa_greedy_partition",
    "py_qa_ising_energy",
    "py_qa_simulated_annealing",
)

DNA_SYMBOLS: tuple[str, ...] = (
    "py_dna_check_cross_hybridization",
    "py_dna_design_orthogonal_set",
    "py_dna_design_sequence",
    "py_dna_detect_hairpins",
    "py_dna_simulate_kinetics",
)

PHOTONIC_SYMBOLS: tuple[str, ...] = (
    "py_ph_analyze_crosstalk",
    "py_ph_analyze_crosstalk_bank",
    "py_ph_analyze_crosstalk_pairs",
)

WORLD_MODEL_SYMBOLS: tuple[str, ...] = ("py_lgssm_kalman_filter",)

PREDICTIVE_CODEC_SYMBOLS: tuple[str, ...] = (
    "py_predict_xor_ema",
    "py_predict_xor_lfsr",
    "py_recover_xor_ema",
    "py_recover_xor_lfsr",
)
