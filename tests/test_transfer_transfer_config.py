# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTransferConfig from former test_transfer.py

"""Focused suite: TestTransferConfig from former test_transfer.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.transfer import (
    apply_transfer_config,
    TransferConfig,
)
from tests.transfer_support import (
    _make_checkpoint,
)


class TestTransferConfig:
    """The transfer-learning configuration."""

    def test_default_config_is_valid(self) -> None:
        """The default configuration freezes nothing."""
        config = TransferConfig()
        assert config.freeze_until == -1
        assert config.lr_backbone == 0.0
        assert config.lr_head == 0.01

    def test_rejects_bool_freeze_until(self) -> None:
        """A boolean freeze index is rejected."""
        with pytest.raises(ValueError, match="freeze_until"):
            TransferConfig(freeze_until=True)

    def test_rejects_negative_freeze_until_below_sentinel(self) -> None:
        """A freeze index below -1 is rejected."""
        with pytest.raises(ValueError, match="freeze_until index"):
            TransferConfig(freeze_until=-2)

    def test_rejects_non_finite_learning_rate(self) -> None:
        """A non-finite learning rate is rejected."""
        with pytest.raises(ValueError, match="learning rates"):
            TransferConfig(lr_head=np.inf)

    def test_rejects_negative_learning_rate(self) -> None:
        """A negative learning rate is rejected."""
        with pytest.raises(ValueError, match="learning rates"):
            TransferConfig(lr_backbone=-0.1)

    def test_apply(self) -> None:
        """Applying gives frozen layers a zero learning rate."""
        c = _make_checkpoint()
        config = TransferConfig(freeze_until=0, lr_backbone=0.0, lr_head=0.01)
        c, lrs = apply_transfer_config(c, config)
        assert lrs[0] == 0.0
        assert lrs[1] == 0.01

    def test_apply_by_name(self) -> None:
        """Applying by name freezes that layer."""
        c = _make_checkpoint()
        config = TransferConfig(freeze_until="hidden", lr_head=0.005)
        c, lrs = apply_transfer_config(c, config)
        assert "hidden" in c.frozen_layers
        assert lrs[1] == 0.005

    def test_no_freeze(self) -> None:
        """Without freezing every layer uses the head rate."""
        c = _make_checkpoint()
        config = TransferConfig(freeze_until=-1, lr_head=0.01)
        c, lrs = apply_transfer_config(c, config)
        assert all(lr == 0.01 for lr in lrs)

    def test_apply_by_unknown_name_rejects_config(self) -> None:
        """Applying an unknown layer name is rejected."""
        c = _make_checkpoint()
        config = TransferConfig(freeze_until="missing-layer", lr_backbone=0.0, lr_head=0.02)
        with pytest.raises(ValueError, match="freeze_until"):
            apply_transfer_config(c, config)
