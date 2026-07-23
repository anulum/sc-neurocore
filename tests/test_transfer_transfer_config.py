# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTransferConfig from former test_transfer.py

"""Focused suite: TestTransferConfig from former test_transfer.py."""

from __future__ import annotations

from tests.transfer_support import *  # noqa: F403

class TestTransferConfig:
    def test_default_config_is_valid(self) -> None:
        config = TransferConfig()
        assert config.freeze_until == -1
        assert config.lr_backbone == 0.0
        assert config.lr_head == 0.01

    def test_rejects_bool_freeze_until(self) -> None:
        with pytest.raises(ValueError, match="freeze_until"):
            TransferConfig(freeze_until=True)

    def test_rejects_negative_freeze_until_below_sentinel(self) -> None:
        with pytest.raises(ValueError, match="freeze_until index"):
            TransferConfig(freeze_until=-2)

    def test_rejects_non_finite_learning_rate(self) -> None:
        with pytest.raises(ValueError, match="learning rates"):
            TransferConfig(lr_head=np.inf)

    def test_rejects_negative_learning_rate(self) -> None:
        with pytest.raises(ValueError, match="learning rates"):
            TransferConfig(lr_backbone=-0.1)

    def test_apply(self) -> None:
        c = _make_checkpoint()
        config = TransferConfig(freeze_until=0, lr_backbone=0.0, lr_head=0.01)
        c, lrs = apply_transfer_config(c, config)
        assert lrs[0] == 0.0
        assert lrs[1] == 0.01

    def test_apply_by_name(self) -> None:
        c = _make_checkpoint()
        config = TransferConfig(freeze_until="hidden", lr_head=0.005)
        c, lrs = apply_transfer_config(c, config)
        assert "hidden" in c.frozen_layers
        assert lrs[1] == 0.005

    def test_no_freeze(self) -> None:
        c = _make_checkpoint()
        config = TransferConfig(freeze_until=-1, lr_head=0.01)
        c, lrs = apply_transfer_config(c, config)
        assert all(lr == 0.01 for lr in lrs)

    def test_apply_by_unknown_name_rejects_config(self) -> None:
        c = _make_checkpoint()
        config = TransferConfig(freeze_until="missing-layer", lr_backbone=0.0, lr_head=0.02)
        with pytest.raises(ValueError, match="freeze_until"):
            apply_transfer_config(c, config)
