# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from sc_neurocore.verification.formal_proofs import FormalVerifier, Interval


def test_interval_addition_and_multiplication_preserve_conservative_bounds():
    left = Interval(-2.0, 3.0)
    right = Interval(4.0, 5.0)

    summed = left + right
    product = left * right

    assert summed == Interval(2.0, 8.0)
    assert product == Interval(-10.0, 15.0)
    assert repr(product) == "[-10.0000, 15.0000]"


def test_probability_bound_verifier_accepts_unit_interval_product():
    assert FormalVerifier.verify_probability_bounds(Interval(0.1, 0.8), Interval(0.0, 1.0))


def test_probability_bound_verifier_rejects_out_of_range_product():
    assert not FormalVerifier.verify_probability_bounds(Interval(0.5, 1.2), Interval(0.5, 1.1))


def test_energy_safety_accepts_non_negative_residual_energy_and_rejects_overdraw():
    assert FormalVerifier.verify_energy_safety(energy=10.0, cost=7.5)
    assert FormalVerifier.verify_energy_safety(energy=3.0, cost=3.0)
    assert not FormalVerifier.verify_energy_safety(energy=2.0, cost=3.0)
