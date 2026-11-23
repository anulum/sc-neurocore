# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIFCellContracts from former test_snn_modules.py

"""Focused suite: TestIFCellContracts from former test_snn_modules.py."""

from __future__ import annotations

import pytest
from tests.test_training.snn_modules_support import (
    IFCell,
    torch,
)


class TestIFCellContracts:
    """Contracts of the integrate-and-fire cell."""

    def test_no_leak_preserves_subthreshold_voltage(self) -> None:
        """Without leak a subthreshold voltage is kept."""
        cell = IFCell(threshold=10.0)

        spike, voltage = cell(torch.tensor([0.0]), torch.tensor([5.0]))

        assert spike.item() == 0.0
        assert voltage.item() == pytest.approx(5.0)

    def test_spike_resets_by_threshold_subtraction(self) -> None:
        """A spike subtracts the threshold from the voltage."""
        cell = IFCell(threshold=1.0)

        spike, voltage = cell(torch.tensor([2.0]), torch.zeros(1))

        assert spike.item() == 1.0
        assert voltage.item() == pytest.approx(1.0)

    def test_surrogate_allows_input_gradient_flow(self) -> None:
        """The surrogate lets gradients reach the input."""
        cell = IFCell()
        current = torch.randn(8, requires_grad=True)

        spike, _ = cell(current, torch.zeros(8))
        spike.sum().backward()

        assert current.grad is not None
