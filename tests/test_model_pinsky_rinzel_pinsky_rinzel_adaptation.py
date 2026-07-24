# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPinskyRinzelAdaptation from former test_model_pinsky_rinzel.py

"""Focused suite: TestPinskyRinzelAdaptation from former test_model_pinsky_rinzel.py."""

from __future__ import annotations

from tests.model_pinsky_rinzel_support import *  # noqa: F403


class TestPinskyRinzelAdaptation:
    def test_isis_lengthen_with_adaptation(self):
        spike_times = _run(PinskyRinzelNeuron(), current_soma=5.0, steps=50000)
        assert len(spike_times) >= 20
        isis = np.diff(spike_times)
        assert np.mean(isis[:5]) <= np.mean(isis[-5:])

    def test_isi_coefficient_of_variation_bounded(self):
        spike_times = _run(PinskyRinzelNeuron(), current_soma=5.0, steps=50000)
        isis = np.diff(spike_times[10:]).astype(float)
        assert np.std(isis) / np.mean(isis) < 0.2
