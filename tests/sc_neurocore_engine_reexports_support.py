# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared engine re-export contract fixtures

"""Shared symbol inventories and availability probes for engine re-export tests."""

from __future__ import annotations

import importlib

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


def _has_inner_symbols(symbols: tuple[str, ...]) -> bool:
    """Return whether the compiled inner module exposes every requested symbol."""
    try:
        inner = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    except ImportError:
        return False
    return all(hasattr(inner, symbol) for symbol in symbols)


def _has_inner_qa() -> bool:
    """Return whether the engine exposes every quantum-annealing binding."""
    return _has_inner_symbols(QA_SYMBOLS)


def _has_inner_dna() -> bool:
    """Return whether the engine exposes every DNA-mapper binding."""
    return _has_inner_symbols(DNA_SYMBOLS)


def _has_inner_photonics() -> bool:
    """Return whether the engine exposes every photonic binding."""
    return _has_inner_symbols(PHOTONIC_SYMBOLS)


def _has_inner_world_model() -> bool:
    """Return whether the engine exposes every world-model binding."""
    return _has_inner_symbols(WORLD_MODEL_SYMBOLS)


def _has_inner_predictive_codec() -> bool:
    """Return whether the engine exposes every predictive-codec binding."""
    return _has_inner_symbols(PREDICTIVE_CODEC_SYMBOLS)
