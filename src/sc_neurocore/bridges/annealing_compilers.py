# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC-to-annealing problem compilers

"""Compile SC network structures into validated Ising and QUBO models."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from sc_neurocore.bridges.annealing_models import IsingModel, QUBOModel


_ZERO_TOLERANCE = 1e-12


def _finite_scalar(name: str, value: float, *, positive: bool = False) -> float:
    """Validate a finite scalar and optionally require positivity."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if positive and numeric <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return numeric


def _square_matrix(name: str, value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return a finite, non-empty square floating-point matrix."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if not bool(np.all(np.isfinite(matrix))):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _labels(node_labels: Sequence[str] | None, size: int) -> list[str]:
    """Return validated unique node labels."""
    if node_labels is None:
        return [f"n{index}" for index in range(size)]
    if isinstance(node_labels, str) or len(node_labels) != size:
        raise ValueError("node_labels must contain exactly one label per node")
    labels = list(node_labels)
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("node_labels must contain non-empty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("node_labels must be unique")
    return labels


class SCToIsing:
    """Compile an SC adjacency matrix into an Ising model."""

    def __init__(
        self,
        coupling_scale: float = 1.0,
        field_scale: float = 0.1,
    ) -> None:
        """Configure finite coupling and local-field scales."""
        self._coupling_scale = _finite_scalar("coupling_scale", coupling_scale)
        self._field_scale = _finite_scalar("field_scale", field_scale)

    def compile(
        self,
        adjacency: np.ndarray[Any, Any],
        node_labels: Sequence[str] | None = None,
        biases: np.ndarray[Any, Any] | None = None,
        name: str = "sc_ising",
    ) -> IsingModel:
        """Compile a finite square weight matrix.

        Directed pairs are averaged. Positive weights become ferromagnetic
        couplings and negative weights become antiferromagnetic couplings.
        """
        matrix = _square_matrix("adjacency", adjacency)
        size = matrix.shape[0]
        labels = _labels(node_labels, size)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")

        if biases is None:
            bias_array = np.zeros(size, dtype=np.float64)
        else:
            bias_array = np.asarray(biases, dtype=np.float64)
            if bias_array.shape != (size,):
                raise ValueError("biases must contain exactly one value per node")
            if not bool(np.all(np.isfinite(bias_array))):
                raise ValueError("biases must contain only finite values")

        fields = {index: float(bias_array[index]) * self._field_scale for index in range(size)}
        couplings: dict[tuple[int, int], float] = {}
        for first in range(size):
            for second in range(first + 1, size):
                weight = float(matrix[first, second] + matrix[second, first]) / 2.0
                if abs(weight) > _ZERO_TOLERANCE:
                    couplings[(first, second)] = -weight * self._coupling_scale

        return IsingModel(
            h=fields,
            J=couplings,
            qubit_labels={index: label for index, label in enumerate(labels)},
            n_qubits=size,
            source=name,
        )


class SCToQUBO:
    """Compile an SC adjacency matrix into a QUBO model."""

    def __init__(self, penalty: float = 2.0) -> None:
        """Configure a positive constraint penalty."""
        self._penalty = _finite_scalar("penalty", penalty, positive=True)

    def compile(
        self,
        adjacency: np.ndarray[Any, Any],
        node_labels: Sequence[str] | None = None,
        name: str = "sc_qubo",
    ) -> QUBOModel:
        """Compile a finite square weight matrix into canonical QUBO terms."""
        matrix = _square_matrix("adjacency", adjacency)
        size = matrix.shape[0]
        labels = _labels(node_labels, size)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")

        q_matrix: dict[tuple[int, int], float] = {}
        for index in range(size):
            q_matrix[(index, index)] = -float(np.sum(np.abs(matrix[:, index])))
            for second in range(index + 1, size):
                weight = float(matrix[index, second] + matrix[second, index]) / 2.0
                if abs(weight) > _ZERO_TOLERANCE:
                    q_matrix[(index, second)] = weight * self._penalty

        return QUBOModel(
            Q=q_matrix,
            qubit_labels={index: label for index, label in enumerate(labels)},
            n_qubits=size,
            source=name,
        )


class SCBitstreamQUBO:
    """Build QUBOs for SC weight selection and connection pruning."""

    def __init__(self, penalty: float = 5.0) -> None:
        """Configure a positive constraint penalty."""
        self._penalty = _finite_scalar("penalty", penalty, positive=True)

    def weight_optimization(
        self,
        target_output: np.ndarray[Any, Any],
        candidate_weights: np.ndarray[Any, Any],
        n_bits: int = 8,
    ) -> QUBOModel:
        """Encode ``||target - candidate_weights @ x||²`` for binary ``x``."""
        target = np.asarray(target_output, dtype=np.float64)
        weights = np.asarray(candidate_weights, dtype=np.float64)
        if target.ndim != 1 or target.size == 0:
            raise ValueError("target_output must be a non-empty one-dimensional array")
        if weights.ndim != 2 or weights.shape[0] != target.shape[0] or weights.shape[1] == 0:
            raise ValueError("candidate_weights must be a non-empty matrix with one row per target")
        if not bool(np.all(np.isfinite(target))) or not bool(np.all(np.isfinite(weights))):
            raise ValueError("weight optimization inputs must contain only finite values")
        if isinstance(n_bits, bool) or not isinstance(n_bits, int) or n_bits <= 0:
            raise ValueError("n_bits must be a positive integer")
        if n_bits > weights.shape[1]:
            raise ValueError("n_bits cannot exceed the number of candidate columns")

        selected_weights = weights[:, :n_bits]
        gram = selected_weights.T @ selected_weights
        correlation = selected_weights.T @ target
        q_matrix: dict[tuple[int, int], float] = {}
        for first in range(n_bits):
            q_matrix[(first, first)] = float(gram[first, first] - 2.0 * correlation[first])
            for second in range(first + 1, n_bits):
                value = float(gram[first, second] + gram[second, first])
                if abs(value) > _ZERO_TOLERANCE:
                    q_matrix[(first, second)] = value

        return QUBOModel(
            Q=q_matrix,
            offset=float(target @ target),
            n_qubits=n_bits,
            source="sc_weight_optimization",
        )

    def pruning(
        self,
        adjacency: np.ndarray[Any, Any],
        importance_scores: np.ndarray[Any, Any],
        max_connections: int,
    ) -> QUBOModel:
        """Select exactly ``max_connections`` undirected candidate edges."""
        matrix = _square_matrix("adjacency", adjacency)
        importance = np.asarray(importance_scores, dtype=np.float64)
        if importance.shape != matrix.shape or not bool(np.all(np.isfinite(importance))):
            raise ValueError("importance_scores must be finite and match adjacency")
        if isinstance(max_connections, bool) or not isinstance(max_connections, int):
            raise ValueError("max_connections must be an integer")

        edges: list[tuple[int, int]] = []
        for first in range(matrix.shape[0]):
            for second in range(first + 1, matrix.shape[0]):
                if max(abs(matrix[first, second]), abs(matrix[second, first])) > _ZERO_TOLERANCE:
                    edges.append((first, second))
        if max_connections < 0 or max_connections > len(edges):
            raise ValueError("max_connections must be between zero and the candidate edge count")

        q_matrix: dict[tuple[int, int], float] = {}
        for edge_index, (first, second) in enumerate(edges):
            symmetric_importance = (
                float(importance[first, second] + importance[second, first]) / 2.0
            )
            q_matrix[(edge_index, edge_index)] = -symmetric_importance

        for first in range(len(edges)):
            q_matrix[(first, first)] += self._penalty * (1 - 2 * max_connections)
            for second in range(first + 1, len(edges)):
                q_matrix[(first, second)] = 2 * self._penalty

        return QUBOModel(
            Q=q_matrix,
            offset=self._penalty * max_connections**2,
            n_qubits=len(edges),
            source="sc_pruning",
        )
