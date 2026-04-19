# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware Constraint Checker

"""Verify that a network respects neuromorphic hardware constraints.

Checks fan-in / fan-out limits, weight precision, delay bounds,
and proposes automatic fixes where possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .device import DeviceSpec


@dataclass
class Violation:
    """A single hardware constraint violation.

    Attributes:
        neuron_id: Index of the offending neuron.
        constraint: Name of the violated constraint.
        value: Actual value that violates the constraint.
        limit: Maximum allowed value.
        message: Human-readable description.
    """

    neuron_id: int
    constraint: str
    value: float
    limit: float
    message: str = ""


@dataclass
class HardwareConstraints:
    """Constraint set for a target device.

    Derived from a ``DeviceSpec``, or specified manually.
    """

    max_fan_in: int = 256
    max_fan_out: int = 4096
    weight_bits: int = 8
    delay_bits: int = 6
    max_delay_ticks: int = 63

    @classmethod
    def from_device(cls, device: DeviceSpec) -> HardwareConstraints:
        """Derive constraints from a device specification."""
        return cls(
            max_fan_in=device.max_fan_in,
            max_fan_out=device.max_fan_out,
            weight_bits=device.weight_bits,
            delay_bits=device.delay_bits,
            max_delay_ticks=device.max_delay_ticks,
        )


class ConstraintChecker:
    """Check and optionally fix hardware constraint violations."""

    def check(
        self,
        adjacency: np.ndarray[Any, Any],
        constraints: HardwareConstraints,
        weights: np.ndarray[Any, Any] | None = None,
        delays: np.ndarray[Any, Any] | None = None,
    ) -> list[Violation]:
        """Check all constraints. Returns list of violations (empty if clean).

        Parameters:
            adjacency: (N, N) connectivity matrix (nonzero = connected).
            constraints: Hardware constraint set.
            weights: Optional (N, N) weight matrix to check precision.
            delays: Optional (N, N) delay matrix in ticks.
        """
        violations: list[Violation] = []
        n = adjacency.shape[0]

        # Fan-in: column sum of binary adjacency
        binary = (adjacency != 0).astype(int)
        fan_in = binary.sum(axis=0)
        for j in range(n):
            if fan_in[j] > constraints.max_fan_in:
                violations.append(
                    Violation(
                        neuron_id=j,
                        constraint="fan_in",
                        value=float(fan_in[j]),
                        limit=float(constraints.max_fan_in),
                        message=f"Neuron {j}: fan-in {fan_in[j]} > {constraints.max_fan_in}",
                    )
                )

        # Fan-out: row sum
        fan_out = binary.sum(axis=1)
        for i in range(n):
            if fan_out[i] > constraints.max_fan_out:
                violations.append(
                    Violation(
                        neuron_id=i,
                        constraint="fan_out",
                        value=float(fan_out[i]),
                        limit=float(constraints.max_fan_out),
                        message=f"Neuron {i}: fan-out {fan_out[i]} > {constraints.max_fan_out}",
                    )
                )

        # Weight precision
        if weights is not None:
            max_abs = np.max(np.abs(weights))
            if max_abs > 0:
                w_max = 2 ** (constraints.weight_bits - 1) - 1
                scale = w_max / max_abs
                quantized = np.round(weights * scale) / scale
                rel_error = np.max(np.abs(weights - quantized)) / max_abs
                if rel_error > 0.1:
                    violations.append(
                        Violation(
                            neuron_id=-1,
                            constraint="weight_precision",
                            value=float(rel_error),
                            limit=0.1,
                            message=f"Weight quantization error {rel_error:.3f} > 10% with {constraints.weight_bits}-bit precision",
                        )
                    )

        # Delay bounds
        if delays is not None:
            max_delay = np.max(delays)
            if max_delay > constraints.max_delay_ticks:
                offenders = np.argwhere(delays > constraints.max_delay_ticks)
                for idx in offenders[:10]:  # report first 10
                    violations.append(
                        Violation(
                            neuron_id=int(idx[0]),
                            constraint="delay",
                            value=float(delays[idx[0], idx[1]]),
                            limit=float(constraints.max_delay_ticks),
                            message=f"Synapse ({idx[0]},{idx[1]}): delay {delays[idx[0], idx[1]]} > {constraints.max_delay_ticks}",
                        )
                    )

        return violations

    def auto_fix(
        self,
        adjacency: np.ndarray[Any, Any],
        constraints: HardwareConstraints,
    ) -> np.ndarray[Any, Any]:
        """Attempt automatic fixes: prune weakest connections to satisfy fan-in/out.

        Returns a modified adjacency matrix.
        """
        adj = adjacency.copy()
        n = adj.shape[0]

        # Fix fan-in violations by pruning weakest incoming connections
        for j in range(n):
            incoming = np.nonzero(adj[:, j])[0]
            if len(incoming) > constraints.max_fan_in:
                strengths = np.abs(adj[incoming, j])
                keep_idx = np.argsort(strengths)[-constraints.max_fan_in :]
                prune_idx = np.setdiff1d(np.arange(len(incoming)), keep_idx)
                for pi in prune_idx:
                    adj[incoming[pi], j] = 0.0

        # Fix fan-out violations by pruning weakest outgoing connections
        for i in range(n):
            outgoing = np.nonzero(adj[i, :])[0]
            if len(outgoing) > constraints.max_fan_out:
                strengths = np.abs(adj[i, outgoing])
                keep_idx = np.argsort(strengths)[-constraints.max_fan_out :]
                prune_idx = np.setdiff1d(np.arange(len(outgoing)), keep_idx)
                for pi in prune_idx:
                    adj[i, outgoing[pi]] = 0.0

        return adj
