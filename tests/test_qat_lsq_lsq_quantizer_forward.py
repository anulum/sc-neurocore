# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLSQQuantizerForward from former test_qat_lsq.py

"""Focused suite: TestLSQQuantizerForward from former test_qat_lsq.py."""

from __future__ import annotations

from tests.qat_lsq_support import *  # noqa: F403


class TestLSQQuantizerForward:
    def test_output_on_integer_grid(self) -> None:
        q = LSQQuantizer(3)  # signed 3-bit: grid [-4, 3]
        x = torch.linspace(-2, 2, 50)
        out = q(x)
        step = q.step.detach()
        codes = torch.round(out / step)
        assert codes.min() >= q.qmin - 1e-6
        assert codes.max() <= q.qmax + 1e-6

    def test_signed_bounds(self) -> None:
        q = LSQQuantizer(4)
        assert q.qmin == -8
        assert q.qmax == 7

    def test_step_initialised_from_first_input(self) -> None:
        q = LSQQuantizer(4)
        assert not bool(q._initialized)
        x = torch.randn(64)
        q(x)
        assert bool(q._initialized)
        expected = 2.0 * x.abs().mean() / math.sqrt(q.qmax)
        assert torch.allclose(q.step.detach(), expected, rtol=1e-5)

    def test_step_not_reinitialised(self) -> None:
        q = LSQQuantizer(4)
        q(torch.randn(64))
        first = q.step.detach().clone()
        q(torch.randn(64) * 100)  # would give a very different init
        assert torch.allclose(q.step.detach(), first)

    def test_rejects_low_bits(self) -> None:
        with pytest.raises(ValueError, match="n_bits must be >= 2"):
            LSQQuantizer(1)

    def test_per_channel_requires_num_channels(self) -> None:
        with pytest.raises(ValueError, match="requires num_channels"):
            LSQQuantizer(4, per_channel=True)
