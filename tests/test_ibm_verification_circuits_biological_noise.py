# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBiologicalNoise from former test_ibm_verification_circuits.py

"""Focused suite: TestBiologicalNoise from former test_ibm_verification_circuits.py."""

from __future__ import annotations

from tests.ibm_verification_circuits_support import *  # noqa: F403


class TestBiologicalNoise:
    def test_noise_params(self):
        ext = pytest.importorskip("posner_extended")
        params = ext.get_noise_params_dict(37.0)
        assert params["temperature_K"] == 310.15
        assert params["T1_nuclear_s"] == 5.0
        assert params["T1_electron_s"] == 1e-6
        assert 0.49 < params["p_excited"] < 0.51
        assert params["cage_dephasing_rate"] is None

    def test_biological_noise_requires_cage_dephasing_rate(self):
        ext = pytest.importorskip("posner_extended")
        pytest.importorskip("qiskit_aer")
        with pytest.raises(ValueError, match="cage_dephasing_rate"):
            ext.biological_noise_model(37.0)
