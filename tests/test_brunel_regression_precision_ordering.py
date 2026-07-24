# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPrecisionOrdering from former test_brunel_regression.py

"""Focused suite: TestPrecisionOrdering from former test_brunel_regression.py."""

from __future__ import annotations

from tests.brunel_regression_support import *  # noqa: F403


class TestPrecisionOrdering:
    """Higher fixed-point precision → more accurate leak → fewer rounding-induced spikes."""

    def test_q16_fewer_spikes_than_q88_short_sim(self):
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0, weight_exc=5.0)
        p3 = translate_v3_fixed_point(bp)
        p11 = translate_v11_q16(bp)

        n3 = FixedPointLIFNeuron(
            data_width=p3["data_width"],
            fraction=p3["fraction"],
            v_threshold=p3["v_threshold_q"],
            v_reset=p3["v_reset_q"],
            refractory_period=p3["refractory_period"],
        )
        n11 = FixedPointLIFNeuron(
            data_width=p11["data_width"],
            fraction=p11["fraction"],
            v_threshold=p11["v_threshold_q"],
            v_reset=p11["v_reset_q"],
            refractory_period=p11["refractory_period"],
        )

        rng = np.random.default_rng(42)
        s3, s11 = 0, 0
        for _ in range(2000):
            I = rng.poisson(200.0 * 0.1 / 1000.0) * int(5.0 * (1 << p3["fraction"])) * 10
            spike3, _ = n3.step(leak_k=p3["leak_k"], gain_k=p3["gain_k"], I_t=I)
            s3 += spike3

            I11 = rng.poisson(200.0 * 0.1 / 1000.0) * int(5.0 * (1 << p11["fraction"])) * 10
            spike11, _ = n11.step(leak_k=p11["leak_k"], gain_k=p11["gain_k"], I_t=I11)
            s11 += spike11

        assert s11 <= s3, f"Q16.12 ({s11}) should produce ≤ spikes than Q8.8 ({s3})"
