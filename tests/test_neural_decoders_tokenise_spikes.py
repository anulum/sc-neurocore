# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTokeniseSpikes from former test_neural_decoders.py

"""Focused suite: TestTokeniseSpikes from former test_neural_decoders.py."""

from __future__ import annotations

from tests.neural_decoders_support import *  # noqa: F403


class TestTokeniseSpikes:
    """Spike tokenisation used by POYO+ and POSSM."""

    def test_empty_input(self) -> None:
        uids, ts = tokenise_spikes([])
        assert len(uids) == 0
        assert len(ts) == 0

    def test_no_spikes(self) -> None:
        trains = [np.zeros(100), np.zeros(100)]
        uids, ts = tokenise_spikes(trains)
        assert len(uids) == 0

    def test_single_spike(self) -> None:
        train = np.zeros(50)
        train[10] = 1
        uids, ts = tokenise_spikes([train], dt=0.5)
        assert len(uids) == 1
        assert uids[0] == 0
        assert ts[0] == pytest.approx(5.0)

    def test_sorted_by_time(self) -> None:
        t0 = np.zeros(100)
        t0[50] = 1
        t1 = np.zeros(100)
        t1[10] = 1
        uids, ts = tokenise_spikes([t0, t1])
        assert ts[0] < ts[1]
        assert uids[0] == 1
        assert uids[1] == 0

    def test_multiple_spikes_per_unit(self) -> None:
        train = np.zeros(20)
        train[5] = 1
        train[15] = 1
        uids, ts = tokenise_spikes([train])
        assert len(uids) == 2
        assert np.all(uids == 0)

    def test_dt_scaling(self) -> None:
        train = np.zeros(10)
        train[4] = 1
        _, ts1 = tokenise_spikes([train], dt=1.0)
        _, ts2 = tokenise_spikes([train], dt=0.1)
        assert ts1[0] == pytest.approx(4.0)
        assert ts2[0] == pytest.approx(0.4)
