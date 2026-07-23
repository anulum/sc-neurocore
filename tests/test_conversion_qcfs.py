# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQCFS from former test_conversion.py

"""Focused suite: TestQCFS from former test_conversion.py."""

from __future__ import annotations

from tests.conversion_support import *  # noqa: F403

class TestQCFS:
    def test_output_range(self) -> None:
        qcfs = QCFSActivation(T=8, theta=1.0)
        x = torch.linspace(-1, 2, 100)
        out = qcfs(x)
        assert out.min() >= 0
        assert out.max() <= 1.0

    def test_monotonic(self) -> None:
        qcfs = QCFSActivation(T=4, theta=1.0)
        x = torch.linspace(0, 1, 100)
        out = qcfs(x)
        diffs = out[1:] - out[:-1]
        assert (diffs >= -1e-6).all(), "QCFS output should be monotonically non-decreasing"

    def test_gradient_flows(self) -> None:
        qcfs = QCFSActivation(T=8, theta=1.0)
        x = torch.tensor([0.5], requires_grad=True)
        out = qcfs(x)
        out.backward()
        assert x.grad is not None

    def test_learnable_theta(self) -> None:
        qcfs = QCFSActivation(T=8, theta=1.0, learn_theta=True)
        assert isinstance(qcfs.theta, nn.Parameter)
