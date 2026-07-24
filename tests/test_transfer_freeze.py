# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFreeze from former test_transfer.py

"""Focused suite: TestFreeze from former test_transfer.py."""

from __future__ import annotations

from tests.transfer_support import *  # noqa: F403


class TestFreeze:
    def test_freeze_by_name(self) -> None:
        c = _make_checkpoint()
        freeze_layers(c, layer_names=["hidden"])
        assert "hidden" in c.frozen_layers
        assert "output" not in c.frozen_layers

    def test_freeze_until_index(self) -> None:
        c = _make_checkpoint()
        freeze_layers(c, until_index=0)
        assert "hidden" in c.frozen_layers
        assert "output" not in c.frozen_layers

    def test_freeze_rejects_unknown_layer(self) -> None:
        c = _make_checkpoint()
        with pytest.raises(ValueError, match="Unknown layer names"):
            freeze_layers(c, layer_names=["missing"])

    def test_freeze_rejects_non_string_layer_name(self) -> None:
        c = _make_checkpoint()
        with pytest.raises(ValueError, match="Layer names must be strings"):
            freeze_layers(c, layer_names=cast(Sequence[str], ["hidden", 1]))

    def test_freeze_rejects_negative_until_index(self) -> None:
        c = _make_checkpoint()
        with pytest.raises(ValueError, match="until_index"):
            freeze_layers(c, until_index=-2)

    def test_freeze_rejects_boolean_until_index(self) -> None:
        c = _make_checkpoint()
        with pytest.raises(ValueError, match="until_index must be an integer"):
            freeze_layers(c, until_index=cast(int, True))

    def test_freeze_rejects_out_of_range_until_index(self) -> None:
        c = _make_checkpoint()
        with pytest.raises(ValueError, match="until_index"):
            freeze_layers(c, until_index=2)

    def test_unfreeze_specific(self) -> None:
        c = _make_checkpoint()
        c.frozen_layers = ["hidden", "output"]
        unfreeze_layers(c, layer_names=["output"])
        assert "hidden" in c.frozen_layers
        assert "output" not in c.frozen_layers

    def test_unfreeze_rejects_unknown_layer(self) -> None:
        c = _make_checkpoint()
        with pytest.raises(ValueError, match="Unknown layer names"):
            unfreeze_layers(c, layer_names=["missing"])

    def test_unfreeze_all(self) -> None:
        c = _make_checkpoint()
        c.frozen_layers = ["hidden", "output"]
        unfreeze_layers(c, all_layers=True)
        assert c.frozen_layers == []

    def test_freeze_layers_deduplicates_and_sorts(self) -> None:
        c = _make_checkpoint()
        c.frozen_layers = ["output"]
        freeze_layers(c, layer_names=["hidden", "output", "hidden"])
        assert c.frozen_layers == ["hidden", "output"]
