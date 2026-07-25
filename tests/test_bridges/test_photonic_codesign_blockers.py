# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Photonic co-design blocker evidence

"""Explicit feasibility blocker evidence for stochastic photonic co-design."""

import numpy as np

from sc_neurocore.bridges import PhotonicCoDesignConfig, StochasticPhotonicCoDesignLoop


def test_codesign_loop_reports_blockers_without_hiding_bad_margins() -> None:
    adjacency = np.ones((4, 4), dtype=np.float64) - np.eye(4, dtype=np.float64)
    config = PhotonicCoDesignConfig(
        bitstream_length=256,
        min_power_margin_db=100.0,
        max_crosstalk_db=-80.0,
        run_fdtd=False,
    )

    report = StochasticPhotonicCoDesignLoop(config).compile(
        adjacency,
        probabilities=[0.2, 0.4, 0.6, 0.8],
    )

    assert report.feasible is False
    assert any("worst optical margin" in blocker for blocker in report.blockers)
    assert any("worst crosstalk" in blocker for blocker in report.blockers)


def test_codesign_loop_flags_density_outside_hoeffding_tolerance() -> None:
    adjacency = np.ones((4, 4), dtype=np.float64) - np.eye(4, dtype=np.float64)
    # A tight density_alpha over a long bitstream shrinks the Hoeffding tolerance
    # below the realised LFSR density error, forcing a density blocker.
    config = PhotonicCoDesignConfig(bitstream_length=4096, density_alpha=0.999999, run_fdtd=False)
    report = StochasticPhotonicCoDesignLoop(config).compile(
        adjacency, probabilities=[0.2, 0.4, 0.6, 0.8]
    )
    assert report.feasible is False
    assert any("Hoeffding tolerance" in blocker for blocker in report.blockers)


def test_codesign_loop_flags_optical_paths_below_detector_margin() -> None:
    # A densely connected 40-node fabric exhausts the WDM split budget, pushing
    # optical paths below the detector margin.
    n = 40
    adjacency = np.ones((n, n), dtype=np.float64) - np.eye(n, dtype=np.float64)
    config = PhotonicCoDesignConfig(bitstream_length=64, run_fdtd=False)
    report = StochasticPhotonicCoDesignLoop(config).compile(adjacency, probabilities=[0.5] * n)
    assert report.feasible is False
    assert any("below detector margin" in blocker for blocker in report.blockers)


def test_codesign_loop_flags_zero_energy_fdtd_pulse() -> None:
    adjacency = np.ones((4, 4), dtype=np.float64) - np.eye(4, dtype=np.float64)
    config = PhotonicCoDesignConfig(bitstream_length=256, run_fdtd=True, fdtd_steps=4)
    # All-zero probabilities encode unmodulated streams, so the representative
    # FDTD pulse carries no field energy.
    report = StochasticPhotonicCoDesignLoop(config).compile(
        adjacency, probabilities=[0.0, 0.0, 0.0, 0.0]
    )
    assert report.feasible is False
    assert any("zero field energy" in blocker for blocker in report.blockers)
