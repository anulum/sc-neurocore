# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stable engine domain-wrapper contracts

"""Verify stable world-model, photonic, DNA, and quantum engine wrappers."""


def test_world_model_wrapper_returns_callable() -> None:
    from sc_neurocore_engine.world_model import get_lgssm_kalman_filter

    assert callable(get_lgssm_kalman_filter())


def test_photonics_wrapper_returns_callable() -> None:
    from sc_neurocore_engine.photonics import (
        get_crosstalk_analyzer,
        get_crosstalk_bank_analyzer,
        get_crosstalk_pair_analyzer,
        has_full_photonic_crosstalk_backend,
    )

    assert callable(get_crosstalk_analyzer())
    assert callable(get_crosstalk_bank_analyzer())
    assert callable(get_crosstalk_pair_analyzer())
    assert has_full_photonic_crosstalk_backend() is True


def test_dna_wrapper_contract_true() -> None:
    from sc_neurocore_engine.dna import has_full_dna_backend

    assert has_full_dna_backend() is True


def test_quantum_wrapper_contract_true() -> None:
    from sc_neurocore_engine.quantum import has_full_quantum_annealing_backend

    assert has_full_quantum_annealing_backend() is True
