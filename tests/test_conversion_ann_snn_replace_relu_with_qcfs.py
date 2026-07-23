# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestReplaceReluWithQcfs from former test_conversion_ann_snn.py

"""Focused suite: TestReplaceReluWithQcfs from former test_conversion_ann_snn.py."""

from __future__ import annotations

from tests.conversion_ann_snn_support import *  # noqa: F403

class TestReplaceReluWithQcfs:
    def test_replaces_relu_in_sequential(self) -> None:
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3), nn.ReLU())
        out = replace_relu_with_qcfs(model, T=6, theta=1.5)
        assert out is model
        activations = [m for m in model if isinstance(m, QCFSActivation)]
        assert len(activations) == 2
        assert not any(isinstance(m, (nn.ReLU, nn.ReLU6)) for m in model)
        assert all(a.T == 6 for a in activations)

    def test_replaces_relu6(self) -> None:
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU6(), nn.Linear(8, 3))
        replace_relu_with_qcfs(model, T=8)
        assert isinstance(model[1], QCFSActivation)

    def test_recurses_into_nested_modules(self) -> None:
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
                self.act = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.act(self.block(x))

        model = Net()
        replace_relu_with_qcfs(model)
        assert isinstance(model.block[1], QCFSActivation)
        assert isinstance(model.act, QCFSActivation)

    def test_learn_theta_propagates(self) -> None:
        model = nn.Sequential(nn.ReLU())
        replace_relu_with_qcfs(model, learn_theta=True)
        assert isinstance(model[0].theta, nn.Parameter)

    def test_default_learn_theta_makes_threshold_trainable(self) -> None:
        model = nn.Sequential(nn.ReLU())
        replace_relu_with_qcfs(model)
        assert isinstance(model[0].theta, nn.Parameter)

    def test_no_relu_leaves_model_unchanged(self) -> None:
        model = nn.Sequential(nn.Linear(4, 3))
        replace_relu_with_qcfs(model)
        assert not any(isinstance(m, QCFSActivation) for m in model.modules())

    def test_requires_torch_when_backend_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sc_neurocore.conversion import ann_to_snn

        monkeypatch.setattr(ann_to_snn, "HAS_TORCH", False)
        with pytest.raises(ImportError, match="PyTorch required"):
            ann_to_snn.replace_relu_with_qcfs(nn.Sequential(nn.ReLU()))
