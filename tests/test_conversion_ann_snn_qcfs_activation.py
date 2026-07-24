# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQCFSActivation from former test_conversion_ann_snn.py

"""Focused suite: TestQCFSActivation from former test_conversion_ann_snn.py."""

from __future__ import annotations

from tests.conversion_ann_snn_support import *  # noqa: F403


class TestQCFSActivation:
    def test_forward(self) -> None:
        from sc_neurocore.conversion.qcfs import QCFSActivation

        act = QCFSActivation(T=8, theta=1.0)
        x = torch.tensor([0.0, 0.5, 1.0, 1.5, -0.5])
        out = act(x)
        assert out[0].item() >= 0
        assert out[-1].item() >= 0  # clamp at 0

    def test_output_range(self) -> None:
        from sc_neurocore.conversion.qcfs import QCFSActivation

        act = QCFSActivation(T=4, theta=1.0)
        x = torch.linspace(-1, 2, 100)
        out = act(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0 + 1e-6

    def test_gradient_flows(self) -> None:
        from sc_neurocore.conversion.qcfs import QCFSActivation

        act = QCFSActivation(T=8, theta=1.0)
        x = torch.tensor([0.5], requires_grad=True)
        out = act(x)
        out.backward()
        assert x.grad is not None

    def test_learnable_theta(self) -> None:
        from sc_neurocore.conversion.qcfs import QCFSActivation

        act = QCFSActivation(T=8, theta=1.0, learn_theta=True)
        assert isinstance(act.theta, nn.Parameter)

    def test_extra_repr(self) -> None:
        from sc_neurocore.conversion.qcfs import QCFSActivation

        act = QCFSActivation(T=8, theta=2.0)
        r = act.extra_repr()
        assert "T=8" in r
        assert "2.00" in r
