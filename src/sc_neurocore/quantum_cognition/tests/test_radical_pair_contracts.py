# SPDX-License-Identifier: AGPL-3.0-or-later
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.

"""Contract tests for radical-pair validation and helper branches."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.quantum_cognition.radical_pair import (
    RadicalPairModel,
    RadicalPairParams,
)


def test_rejects_invalid_quadrature_order() -> None:
    """Constructor rejects quadrature rules that cannot integrate an interval."""
    with pytest.raises(ValueError, match="quadrature_order"):
        RadicalPairModel(RadicalPairParams(quadrature_order=1))


def test_explicit_hyperfine_tensor_state_counts() -> None:
    """Explicit tensor construction records nuclei on both radicals."""
    model = RadicalPairModel.from_hyperfine_tensors(
        tensors_1=[np.eye(3, dtype=np.float64)],
        tensors_2=[np.eye(3, dtype=np.float64) * 2.0],
        exchange_j=0.25,
        recombination_rate=0.5,
        lifetime_us=2.0,
        quadrature_order=4,
    )

    state = model.get_state()
    assert state["hyperfine_a_mhz"] == 0.0
    assert state["exchange_j_mhz"] == 0.25
    assert state["n_hyperfine_tensors_1"] == 1
    assert state["n_hyperfine_tensors_2"] == 1
    assert "nuclei=2" in repr(model)


def test_invalid_hyperfine_tensor_shape_is_rejected() -> None:
    """Tensor validation reports the exact radical-side tensor index."""
    model = RadicalPairModel.from_hyperfine_tensors(
        tensors_1=[np.ones((2, 2), dtype=np.float64)],
        tensors_2=[],
        exchange_j=1.0,
        recombination_rate=0.1,
        lifetime_us=1.0,
        quadrature_order=2,
    )

    with pytest.raises(ValueError, match=r"hyperfine_tensors_1\[0\].*shape"):
        model.singlet_yield(0.0)


def test_zero_nucleus_singlet_density_has_electron_dimension() -> None:
    """The no-bath helper returns pure electron singlet density matrices."""
    rho, projector = RadicalPairModel._singlet_density_with_nuclear_bath(0)

    assert rho.shape == (4, 4)
    assert projector.shape == (4, 4)
    assert np.trace(rho) == pytest.approx(1.0)
    assert np.allclose(rho, projector)


def test_dense_hamiltonian_rejects_oversized_nuclear_bath() -> None:
    """Dense exact evolution fail-closes before allocating huge spin operators."""
    tensors = [np.eye(3, dtype=np.float64) for _ in range(9)]
    model = RadicalPairModel.from_hyperfine_tensors(
        tensors_1=tensors,
        tensors_2=[],
        exchange_j=1.0,
        recombination_rate=0.1,
        lifetime_us=1.0,
        quadrature_order=2,
    )

    with pytest.raises(ValueError, match="supports up to 8 nuclei"):
        model.singlet_yield(0.0)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (RadicalPairParams(recombination_rate=0.0), "recombination_rate"),
        (RadicalPairParams(lifetime_us=0.0), "lifetime_us"),
    ],
)
def test_singlet_yield_rejects_non_positive_rates(
    params: RadicalPairParams,
    message: str,
) -> None:
    """Singlet-yield validation rejects non-positive kinetic parameters."""
    with pytest.raises(ValueError, match=message):
        RadicalPairModel(params).singlet_yield(0.0)
