# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCanFoldHeterogeneity from former test_folded_heterogeneous_params.py

"""Focused suite: TestCanFoldHeterogeneity from former test_folded_heterogeneous_params.py."""

from __future__ import annotations

from tests.folded_heterogeneous_params_support import *  # noqa: F403


class TestCanFoldHeterogeneity:
    """``_can_fold`` accepts heterogeneous datapath parameters."""

    def test_homogeneous_graph_folds(self) -> None:
        assert _can_fold(_qgraph([10.0, 10.0, 10.0]), data_width=16) is True

    def test_heterogeneous_graph_folds(self) -> None:
        assert _can_fold(_qgraph([10.0, 20.0, 30.0]), data_width=16) is True
