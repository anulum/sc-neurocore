# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPopulationDecode from former test_utils.py

"""Focused suite: TestPopulationDecode from former test_utils.py."""

from __future__ import annotations

from tests.test_training.utils_support import *  # noqa: F403


class TestPopulationDecode:
    """Tests for population-vector decoding contracts."""

    def test_argmax_equivalent(self) -> None:
        """With one-hot spike counts, should recover the index."""
        counts = torch.tensor([[0.0, 0.0, 5.0, 0.0]])
        decoded = population_decode(counts)
        assert decoded.item() == pytest.approx(2.0)

    def test_weighted_average(self) -> None:
        """Uniform two-neuron activity decodes to the midpoint index."""
        counts = torch.tensor([[1.0, 1.0]])
        decoded = population_decode(counts)
        assert decoded.item() == pytest.approx(0.5)  # mean of 0 and 1

    def test_custom_preferred_values(self) -> None:
        """Custom one-dimensional preferred values override neuron indices."""
        counts = torch.tensor([[1.0, 0.0, 0.0]])
        preferred = torch.tensor([10.0, 20.0, 30.0])
        decoded = population_decode(counts, preferred)
        assert decoded.item() == pytest.approx(10.0)

    def test_batch(self) -> None:
        """Batch decoding returns one value per sample."""
        counts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        decoded = population_decode(counts)
        assert decoded.shape == (2,)
        assert decoded[0].item() == pytest.approx(0.0)
        assert decoded[1].item() == pytest.approx(1.0)

    def test_multidim_preferred(self) -> None:
        """Multi-dimensional preferred values decode to vector outputs."""
        counts = torch.tensor([[1.0, 1.0]])
        preferred = torch.tensor([[0.0, 0.0], [2.0, 4.0]])
        decoded = population_decode(counts, preferred)
        assert decoded.shape == (1, 2)
        assert decoded[0, 0].item() == pytest.approx(1.0)
        assert decoded[0, 1].item() == pytest.approx(2.0)
