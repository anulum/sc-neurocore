# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN bridge source-mode contracts

from __future__ import annotations


import numpy as np
import pytest

from scpn_neurocore.bridge import (
    SourceDataUnavailable,
    load_connectome,
    load_live_stream,
    load_power_grid,
    load_tokamak_data,
)
from tests.scpn_neurocore_bridge_support import assert_qpu_artifact


def test_default_loaders_do_not_silently_generate_data() -> None:
    with pytest.raises(SourceDataUnavailable, match="source_mode"):
        load_connectome("c_elegans_sub", n=14)

    with pytest.raises(SourceDataUnavailable, match="source_mode"):
        load_tokamak_data()

    with pytest.raises(SourceDataUnavailable, match="source_mode"):
        load_power_grid(16)

    with pytest.raises(SourceDataUnavailable, match="source_mode"):
        load_live_stream(source="eeg_powergrid", step=0)


def test_publication_source_modes_raise_when_source_is_missing() -> None:
    with pytest.raises(SourceDataUnavailable, match="c_elegans_sub"):
        load_connectome("c_elegans_sub", n=14, source_mode="curated")

    with pytest.raises(SourceDataUnavailable, match="tokamak"):
        load_tokamak_data(source_mode="recorded")


def test_synthetic_connectome_artifact_is_labelled_and_valid() -> None:
    artifact = load_connectome("c_elegans_sub", n=14, source_mode="synthetic")

    assert_qpu_artifact(artifact, 14, "synthetic")
    assert artifact.domain == "connectome"
    assert artifact.metadata["publication_safe"] is False


def test_synthetic_tokamak_artifact_is_qpu_ready() -> None:
    artifact = load_tokamak_data(n=16, synthetic=True)

    assert_qpu_artifact(artifact, 16, "synthetic")
    assert artifact.domain == "tokamak"


def test_synthetic_power_grid_artifact_supports_campaign_sizes() -> None:
    for n in (16, 20):
        artifact = load_power_grid(n=n, name="power_grid_europe", source_mode="fixture")
        assert_qpu_artifact(artifact, n, "fixture")
        assert artifact.domain == "power_grid"


def test_synthetic_live_stream_is_replayable_per_step() -> None:
    a = load_live_stream(source="eeg_powergrid", step=3, source_mode="synthetic")
    b = load_live_stream(source="eeg_powergrid", step=3, source_mode="synthetic")
    c = load_live_stream(source="eeg_powergrid", step=4, source_mode="synthetic")

    assert_qpu_artifact(a, 12, "synthetic")
    np.testing.assert_allclose(a.K_nm, b.K_nm)
    np.testing.assert_allclose(a.omega, b.omega)
    assert not np.allclose(a.omega, c.omega)
    assert a.replay_id == "synthetic:eeg_powergrid:step:3"


def test_bridge_rejects_invalid_source_inputs() -> None:
    with pytest.raises(ValueError, match="unsupported connectome"):
        load_connectome("unknown", source_mode="synthetic")

    with pytest.raises(ValueError, match="unsupported live stream"):
        load_live_stream(source="unknown", step=0, source_mode="synthetic")

    with pytest.raises(ValueError, match="step"):
        load_live_stream(source="eeg_powergrid", step=-1, source_mode="synthetic")
