# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConstraintChecker from former test_hardware.py

"""Focused suite: TestConstraintChecker from former test_hardware.py."""

from __future__ import annotations

from tests.hardware_support import *  # noqa: F403

class TestConstraintChecker:
    def test_no_violations_small_network(self):
        adj = np.zeros((5, 5))
        adj[0, 1] = 1.0
        adj[1, 2] = 1.0
        checker = ConstraintChecker()
        constraints = HardwareConstraints(max_fan_in=256, max_fan_out=4096)
        violations = checker.check(adj, constraints)
        assert len(violations) == 0

    def test_fan_in_violation(self):
        n = 50
        adj = np.zeros((n, n))
        adj[:, 0] = 1.0  # neuron 0 has fan-in = 49
        adj[0, 0] = 0
        checker = ConstraintChecker()
        constraints = HardwareConstraints(max_fan_in=10)
        violations = checker.check(adj, constraints)
        fan_in_violations = [v for v in violations if v.constraint == "fan_in"]
        assert len(fan_in_violations) >= 1
        assert fan_in_violations[0].neuron_id == 0

    def test_fan_out_violation(self):
        n = 50
        adj = np.zeros((n, n))
        adj[0, :] = 1.0
        adj[0, 0] = 0
        checker = ConstraintChecker()
        constraints = HardwareConstraints(max_fan_out=10)
        violations = checker.check(adj, constraints)
        fan_out_violations = [v for v in violations if v.constraint == "fan_out"]
        assert len(fan_out_violations) >= 1

    def test_delay_violation(self):
        adj = np.array([[0, 1], [0, 0]], dtype=float)
        delays = np.array([[0, 100], [0, 0]], dtype=float)
        checker = ConstraintChecker()
        constraints = HardwareConstraints(max_delay_ticks=63)
        violations = checker.check(adj, constraints, delays=delays)
        delay_v = [v for v in violations if v.constraint == "delay"]
        assert len(delay_v) >= 1

    def test_weight_precision_violation(self):
        checker = ConstraintChecker()
        constraints = HardwareConstraints(weight_bits=2)
        weights = np.array([[0.0, 0.49], [1.0, 0.0]], dtype=float)

        violations = checker.check(np.ones((2, 2)), constraints, weights=weights)

        weight_v = [v for v in violations if v.constraint == "weight_precision"]
        assert len(weight_v) == 1
        assert weight_v[0].neuron_id == -1
        assert weight_v[0].value > weight_v[0].limit

    def test_zero_weights_skip_precision_violation(self):
        checker = ConstraintChecker()
        constraints = HardwareConstraints(weight_bits=1)

        violations = checker.check(np.ones((2, 2)), constraints, weights=np.zeros((2, 2)))

        assert [v for v in violations if v.constraint == "weight_precision"] == []

    def test_from_device_constraints(self):
        constraints = HardwareConstraints.from_device(get_device(DeviceFamily.LOIHI))
        assert constraints.max_fan_in == 4096
        assert constraints.weight_bits == 9

    def test_auto_fix_resolves_violations(self):
        n = 50
        adj = np.zeros((n, n))
        adj[:, 0] = 1.0
        adj[0, 0] = 0
        checker = ConstraintChecker()
        constraints = HardwareConstraints(max_fan_in=10)
        violations_before = checker.check(adj, constraints)
        assert len(violations_before) > 0

        fixed = checker.auto_fix(adj, constraints)
        violations_after = checker.check(fixed, constraints)
        fan_in_after = [v for v in violations_after if v.constraint == "fan_in"]
        assert len(fan_in_after) == 0

    def test_auto_fix_resolves_fan_out_violations(self):
        n = 50
        adj = np.zeros((n, n))
        adj[0, :] = np.linspace(1.0, 50.0, n)
        adj[0, 0] = 0.0
        checker = ConstraintChecker()
        constraints = HardwareConstraints(max_fan_out=10)

        fixed = checker.auto_fix(adj, constraints)

        violations_after = checker.check(fixed, constraints)
        fan_out_after = [v for v in violations_after if v.constraint == "fan_out"]
        assert fan_out_after == []
        assert np.count_nonzero(fixed[0, :]) == 10
