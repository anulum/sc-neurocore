# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestValidation from former test_sc_forward.py

"""Focused suite: TestValidation from former test_sc_forward.py."""

from __future__ import annotations

from tests.sc_forward_support import *  # noqa: F403


class TestValidation:
    """Shape and range validation of the NumPy reference."""

    def test_non_positive_length(self) -> None:
        with pytest.raises(ValueError, match="length must be positive"):
            sc_forward_numpy(np.zeros((1, 1, 1), dtype=np.uint64), np.zeros(1), 0)

    def test_weights_not_3d(self) -> None:
        with pytest.raises(ValueError, match="must be 3-D"):
            sc_forward_numpy(np.zeros((1, 1), dtype=np.uint64), np.zeros(1), 64)

    def test_word_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="ceil"):
            sc_forward_numpy(np.zeros((1, 1, 5), dtype=np.uint64), np.zeros(1), 64)

    def test_input_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="length n_in"):
            sc_forward_numpy(np.zeros((1, 2, 1), dtype=np.uint64), np.zeros(3), 64)

    def test_probability_out_of_range(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            sc_forward_numpy(np.zeros((1, 1, 1), dtype=np.uint64), np.array([1.5]), 64)
