# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum-annealing input-scalar validation tests

"""Hold every annealing input check to one admission table and one wording."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from sc_neurocore.bridges.annealing_analysis import TTSAnalyzer
from sc_neurocore.bridges.annealing_compilers import SCBitstreamQUBO, SCToIsing, SCToQUBO
from sc_neurocore.bridges.annealing_decomposition import ProblemDecomposer
from sc_neurocore.bridges.annealing_models import CouplerSpec, IsingModel, QUBOModel, QubitSpec
from sc_neurocore.bridges.annealing_solvers import SimulatedAnnealer
from sc_neurocore.bridges.annealing_transforms import AnnealingSchedule, SCPrecisionEncoder
from sc_neurocore.bridges.annealing_validation import (
    finite_real,
    non_negative_index,
    positive_int,
    positive_real,
)
from tests.test_bridges.quantum_annealing_test_helpers import unsafe

REAL_ACCEPTED = (
    (3, 3.0),
    (2.5, 2.5),
    (np.int64(4), 4.0),
    (np.float32(0.5), 0.5),
    (np.float64(1.25), 1.25),
    (Fraction(3, 4), 0.75),
)
REAL_REFUSED = (True, False, np.bool_(True), Decimal("1.5"), "1.5", b"1", np.array(1.0), None, 1j)
INTEGER_REFUSED = (True, np.bool_(False), 2.0, np.float64(2.0), Fraction(2, 1), "2", None)


def _refused(call: Callable[[], object], message: str) -> None:
    """Require a ValueError whose message contains ``message``."""
    with pytest.raises(ValueError, match=message):
        call()


@pytest.mark.parametrize(("value", "expected"), REAL_ACCEPTED)
def test_real_helpers_accept_python_numpy_and_fraction_values(
    value: object, expected: float
) -> None:
    """Python and NumPy integers and floats and fractions convert to floats."""
    assert finite_real("x", value) == expected
    assert type(finite_real("x", value)) is float
    assert positive_real("x", value) == expected


@pytest.mark.parametrize("value", REAL_REFUSED)
def test_real_helpers_refuse_booleans_and_non_real_values(value: object) -> None:
    """Booleans, decimals, strings, bytes, arrays, None and complex are not real numbers."""
    _refused(lambda: finite_real("gain", value), "gain must be a real number")
    _refused(lambda: positive_real("gain", value), "gain must be a real number")


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), np.float32("inf"), 10**400]
)
def test_real_helpers_refuse_values_without_a_finite_float(value: object) -> None:
    """NaN, infinities and integers beyond the float range are not finite."""
    _refused(lambda: finite_real("gain", value), "gain must be finite")


@pytest.mark.parametrize("value", [0, 0.0, -1, np.float64(-0.5)])
def test_positive_real_refuses_zero_and_negative_values(value: object) -> None:
    """Positivity is checked after the type and finiteness."""
    _refused(lambda: positive_real("rate", value), "rate must be greater than zero")
    assert finite_real("rate", value) <= 0.0


@pytest.mark.parametrize(("value", "expected"), [(1, 1), (np.int64(7), 7), (np.uint8(3), 3)])
def test_integer_helpers_accept_python_and_numpy_integers(value: object, expected: int) -> None:
    """Python and NumPy integers convert to Python integers."""
    assert positive_int("n", value) == expected
    assert type(positive_int("n", value)) is int
    assert non_negative_index("i", value) == expected


@pytest.mark.parametrize("value", INTEGER_REFUSED)
def test_integer_helpers_refuse_booleans_and_non_integers(value: object) -> None:
    """Booleans and every non-integral value are refused, integral floats included."""
    _refused(lambda: positive_int("n", value), "n must be a positive integer")
    _refused(lambda: non_negative_index("i", value), "i must be a non-negative integer")


def test_integer_helpers_bound_zero_and_negative_values() -> None:
    """Zero is an index but not a positive count; negatives are neither."""
    assert non_negative_index("i", 0) == 0
    _refused(lambda: positive_int("n", 0), "n must be a positive integer")
    _refused(lambda: positive_int("n", np.int64(-2)), "n must be a positive integer")
    _refused(lambda: non_negative_index("i", -1), "i must be a non-negative integer")


@pytest.mark.parametrize("bad", [True, "1.5"])
@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda bad: IsingModel(h={0: bad}, n_qubits=1), r"h\[0\] must be a real number"),
        (
            lambda bad: IsingModel(J={(0, 1): bad}, n_qubits=2),
            r"J\[\(0, 1\)\] must be a real number",
        ),
        (
            lambda bad: QUBOModel(Q={(0, 0): bad}, n_qubits=1),
            r"Q\[\(0, 0\)\] must be a real number",
        ),
        (lambda bad: QubitSpec(index=0, label="q0", bias=bad), "bias must be a real number"),
        (
            lambda bad: CouplerSpec(qubit_a=0, qubit_b=1, strength=bad),
            "strength must be a real number",
        ),
        (lambda bad: SCToIsing(coupling_scale=bad), "coupling_scale must be a real number"),
        (lambda bad: SCToQUBO(penalty=bad), "penalty must be a real number"),
        (lambda bad: SCBitstreamQUBO(penalty=bad), "penalty must be a real number"),
        (lambda bad: SimulatedAnnealer(beta_start=bad), "beta_start must be a real number"),
        (lambda bad: AnnealingSchedule().linear(bad), "duration_us must be a real number"),
        (lambda bad: SCPrecisionEncoder().encode(bad), "sc_value must be a real number"),
        (lambda bad: TTSAnalyzer().compute(bad, 20.0), "p_success must be a real number"),
    ],
)
def test_every_caller_refuses_booleans_and_strings(
    build: Callable[[object], object], message: str, bad: object
) -> None:
    """Each module that used a loose float() check now refuses booleans and strings."""
    _refused(lambda: build(bad), message)


def test_integer_fields_refuse_booleans_at_their_constructors() -> None:
    """Counts and indices refuse ``True`` at the public entry points."""
    _refused(lambda: SimulatedAnnealer(n_sweeps=True), "n_sweeps must be a positive integer")
    _refused(lambda: ProblemDecomposer(max_subproblem_size=True), "max_subproblem_size")
    _refused(lambda: QubitSpec(index=True, label="q"), "index must be a non-negative integer")


def test_numpy_scalars_are_accepted_through_public_constructors() -> None:
    """Coefficients and counts taken from NumPy arrays are admitted and normalised.

    The public annotations stay ``float`` and ``int``, so the NumPy scalars pass
    through ``unsafe`` for the type checker; at run time they are accepted.
    """
    biases = np.array([2, -1], dtype=np.int64)
    model = IsingModel(
        h={0: biases[0], 1: biases[1]}, J={(0, 1): unsafe(np.float32(0.5))}, n_qubits=2
    )
    assert model.h == {0: 2.0, 1: -1.0}
    assert model.J == {(0, 1): 0.5}
    assert all(type(value) is float for value in [*model.h.values(), *model.J.values()])
    weights = np.array([[0.0, 1.0], [0.0, 0.0]])
    from_numpy = SCToIsing(
        coupling_scale=unsafe(np.float32(0.5)), field_scale=unsafe(np.int64(2))
    ).compile(weights)
    from_python = SCToIsing(coupling_scale=0.5, field_scale=2.0).compile(weights)
    assert from_numpy.J == from_python.J
    assert from_numpy.h == from_python.h
    decomposer = ProblemDecomposer(max_subproblem_size=unsafe(np.int64(4)), overlap=0)
    assert decomposer.decompose(model) == [model]
    assert TTSAnalyzer().compute(np.float64(0.5), unsafe(np.int64(20)))["tts_us"] > 0.0
