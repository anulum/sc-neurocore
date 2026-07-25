# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Photonic co-design validation contracts

"""Compile-input and configuration validation for photonic co-design."""

import numpy as np
import pytest

from sc_neurocore.bridges import PhotonicCoDesignConfig, StochasticPhotonicCoDesignLoop


def test_codesign_loop_rejects_mismatched_inputs() -> None:
    loop = StochasticPhotonicCoDesignLoop(PhotonicCoDesignConfig(run_fdtd=False))
    with pytest.raises(ValueError, match="at least one node"):
        loop.compile(np.zeros((0, 0), dtype=np.float64))
    with pytest.raises(ValueError, match="node_labels"):
        loop.compile(np.eye(2), node_labels=["only_one"])
    with pytest.raises(ValueError, match="probabilities"):
        loop.compile(np.eye(2), probabilities=[0.5, 0.5, 0.5])


def test_config_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="bitstream_length must be positive"):
        PhotonicCoDesignConfig(bitstream_length=0)
    with pytest.raises(ValueError, match="density_alpha"):
        PhotonicCoDesignConfig(density_alpha=1.0)
    with pytest.raises(ValueError, match="fdtd_steps"):
        PhotonicCoDesignConfig(fdtd_steps=-1)


def test_codesign_loop_rejects_non_square_adjacency() -> None:
    loop = StochasticPhotonicCoDesignLoop(PhotonicCoDesignConfig(run_fdtd=False))
    with pytest.raises(ValueError, match="square matrix"):
        loop.compile(np.zeros((2, 3), dtype=np.float64))
