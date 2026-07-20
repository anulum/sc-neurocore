# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Aging-reliability contracts

"""Contracts for compiler aging and reliability prediction."""

from __future__ import annotations


class TestReliability:
    def test_nominal(self) -> None:
        from sc_neurocore.compiler.intelligence import predict_reliability

        r = predict_reliability(voltage_v=0.9, temperature_c=25.0)
        assert r.mttf_years > 0
        assert r.failure_mode == min(
            r.mechanism_mttf_hours,
            key=lambda name: r.mechanism_mttf_hours[name],
        )

    def test_high_temp(self) -> None:
        from sc_neurocore.compiler.intelligence import predict_reliability

        r = predict_reliability(temperature_c=125.0)
        assert r.failure_mode == min(
            r.mechanism_mttf_hours,
            key=lambda name: r.mechanism_mttf_hours[name],
        )
        assert r.temp_accel > 1.0
