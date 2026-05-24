# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for identity checkpoint contracts

"""Contracts for identity substrate checkpoint save/load and merge boundaries."""

from __future__ import annotations

import pytest

from sc_neurocore.identity.checkpoint import Checkpoint
from sc_neurocore.identity.substrate import IdentitySubstrate


def test_checkpoint_round_trips_identity_substrate(tmp_path) -> None:
    substrate = IdentitySubstrate(n_cortical=8)
    path = tmp_path / "identity.npz"

    Checkpoint.save(substrate, path)
    restored = Checkpoint.load(str(path))

    assert restored.n_cortical == 8


def test_checkpoint_merge_single_path_returns_loaded_substrate(tmp_path) -> None:
    substrate = IdentitySubstrate(n_cortical=8)
    path = tmp_path / "single.npz"
    Checkpoint.save(substrate, path)

    merged = Checkpoint.merge([str(path)])

    assert merged.n_cortical == 8


def test_checkpoint_merge_rejects_empty_path_list() -> None:
    with pytest.raises(ValueError):
        Checkpoint.merge([])
