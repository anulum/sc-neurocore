# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFloatToBipolarWeights from former test_bipolar_sc.py

"""Focused suite: TestFloatToBipolarWeights from former test_bipolar_sc.py."""

from __future__ import annotations

from tests.bipolar_sc_support import *  # noqa: F403

class TestFloatToBipolarWeights:
    def test_normalises_to_minus_one_one(self):
        w = np.array([[-2.0, 1.0], [0.5, -0.3]])
        bp = float_to_bipolar_weights(w)
        assert bp.max() <= 1.0
        assert bp.min() >= -1.0
        assert abs(bp.max() - 1.0) < 1e-6 or abs(bp.min() + 1.0) < 1e-6

    def test_preserves_sign(self):
        w = np.array([-3.0, 2.0, 0.0, -1.0])
        bp = float_to_bipolar_weights(w)
        assert bp[0] < 0
        assert bp[1] > 0
        assert bp[2] == 0.0

    def test_torch_tensor(self):
        torch = __import__("pytest").importorskip("torch")
        w = torch.tensor([[-1.5, 0.5], [0.3, -0.8]])
        bp = float_to_bipolar_weights(w)
        assert isinstance(bp, np.ndarray)
        assert bp.shape == (2, 2)

    def test_rejects_non_finite_weights(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            float_to_bipolar_weights(np.array([1.0, np.inf]))
