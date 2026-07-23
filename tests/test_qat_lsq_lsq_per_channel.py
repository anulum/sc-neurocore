# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLSQPerChannel from former test_qat_lsq.py

"""Focused suite: TestLSQPerChannel from former test_qat_lsq.py."""

from __future__ import annotations

from tests.qat_lsq_support import *  # noqa: F403

class TestLSQPerChannel:
    def test_per_channel_step_shape(self) -> None:
        q = LSQQuantizer(4, per_channel=True, num_channels=5)
        assert q.step.shape == torch.Size([5])

    def test_per_channel_steps_are_independent(self) -> None:
        q = LSQQuantizer(4, per_channel=True, ch_axis=0, num_channels=3)
        # Row magnitudes differ by 100x -> per-channel steps must differ.
        w = torch.stack([torch.ones(8) * 0.01, torch.ones(8), torch.ones(8) * 100.0])
        q(w)
        steps = q.step.detach()
        assert steps[2] > steps[1] > steps[0]

    def test_per_tensor_step_is_scalar(self) -> None:
        q = LSQQuantizer(4)
        q(torch.randn(10))
        assert q.step.shape == torch.Size([])
