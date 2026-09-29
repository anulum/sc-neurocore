# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Coupled dense IF parameter admission through the public network

"""Exercise parameter snapshots and refusals through the actual public network."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.conversion import ConvertedSNN
from sc_neurocore.conversion.if_parameters import OutputMode


@pytest.mark.parametrize(
    "weights,biases,thresholds",
    [
        ([], [], []),
        ([[[1.0]]], [], [1.0]),
        ([[[1.0]]], [None], []),
        ([[1.0]], [None], [1.0]),
        ([np.empty((0, 1))], [None], [1.0]),
        ([[[1.0]], [[1.0, 1.0]]], [None, None], [1.0, 1.0]),
        ([[[1.0]]], [[0.0, 0.0]], [1.0]),
        ([[[1.0]]], [None], [0.0]),
        ([[[1.0]]], [None], [-1.0]),
        ([[[1.0]]], [None], [float("inf")]),
        ([[[1.0]]], [None], [float("nan")]),
        ([[[float("inf")]]], [None], [1.0]),
        ([[[1.0]]], [[float("nan")]], [1.0]),
        ([[[1.0 + 1.0j]]], [None], [1.0]),
        ([[["1"]]], [None], [1.0]),
    ],
)
def test_invalid_coupled_parameters_refused(
    weights: Sequence[npt.ArrayLike],
    biases: Sequence[npt.ArrayLike | None],
    thresholds: Sequence[float],
) -> None:
    """Require real finite coefficients with complete connected layer geometry."""
    with pytest.raises(ValueError):
        ConvertedSNN(weights, biases, thresholds, T=2)


@pytest.mark.parametrize("steps", [0, -1, True, 1.5, 2**53 + 1])
def test_invalid_step_budget_refused(steps: int) -> None:
    """Keep event counts inside the exactly representable float64 integer domain."""
    with pytest.raises(ValueError, match="positive integer"):
        ConvertedSNN([[[1.0]]], [None], [1.0], T=steps)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_output_scale_refused(scale: float) -> None:
    """Require a finite positive conversion back to source activation units."""
    with pytest.raises(ValueError, match="output scale"):
        ConvertedSNN([[[1.0]]], [None], [1.0], T=1, output_scale=scale)


@pytest.mark.parametrize("shift", [float("nan"), float("inf")])
def test_invalid_initial_fraction_refused(shift: float) -> None:
    """Reject nonfinite default membrane states before any simulation."""
    with pytest.raises(ValueError, match="initial membrane fraction"):
        ConvertedSNN([[[1.0]]], [None], [1.0], T=1, initial_membrane_fraction=shift)


@pytest.mark.parametrize("mode", ["unknown", "", "Linear"])
def test_unknown_output_mode_refused(mode: OutputMode) -> None:
    """Admit only the two declared final-layer dynamical responses."""
    with pytest.raises(ValueError, match="output mode"):
        ConvertedSNN([[[1.0]]], [None], [1.0], T=1, output_mode=mode)


def test_constructor_owns_parameters_and_replay_revalidates_edits() -> None:
    """Prevent external array changes or later public edits from corrupting a replay."""
    weight = np.array([[1.0]])
    bias = np.array([0.0])
    model = ConvertedSNN([weight], [bias], [1.0], T=1)
    weight.fill(100.0)
    bias.fill(100.0)
    np.testing.assert_array_equal(model.run([1.0]), [1.0])
    model.weights[0][0, 0] = float("nan")
    with pytest.raises(ValueError, match="weight must be finite"):
        model.replay(np.ones((1, 1, 1)))


def test_wider_float_parameter_conversion_has_documented_refusal() -> None:
    """Report a real representation overflow as the parameter admission error."""
    values = np.array([[np.finfo(np.longdouble).max]], dtype=np.longdouble)
    if np.finfo(np.longdouble).max > np.finfo(np.float64).max:
        with pytest.raises(ValueError, match="represented in float64"):
            ConvertedSNN([values], [None], [1.0], T=1)
    else:
        network = ConvertedSNN([values], [None], [1.0], T=1)
        np.testing.assert_array_equal(network.weights[0], values)


@pytest.mark.parametrize("fractions", [[], [0.0, 0.5], [float("nan")], [float("inf")]])
def test_invalid_per_layer_preloads_refused(fractions: list[float]) -> None:
    """Require exactly one finite preload for each weighted stage."""
    with pytest.raises(ValueError, match="layer membrane fractions"):
        ConvertedSNN([[[1.0]]], [None], [1.0], T=1, layer_membrane_fractions=fractions)


def test_per_layer_preload_input_is_owned_and_late_edits_are_revalidated() -> None:
    """Keep caller preload lists independent and reject invalid object-field edits."""
    fractions = [0.5]
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=1, layer_membrane_fractions=fractions)
    fractions[0] = 0.0
    result = model.replay(np.array([[[0.5]]]), binary_inputs=False)
    np.testing.assert_array_equal(result.output, [[1.0]])
    assert model.layer_membrane_fractions is not None
    model.layer_membrane_fractions[0] = float("nan")
    with pytest.raises(ValueError, match="layer membrane fractions"):
        model.run([1.0])


@pytest.mark.parametrize("field", ["threshold", "global", "layer", "scale"])
@pytest.mark.parametrize("value", [np.finfo(np.longdouble).max])
def test_extended_scalar_cannot_bypass_float64_parameter_domain(field: str, value: float) -> None:
    """Finite wider-format scalars cannot become nonfinite native coefficients."""
    thresholds = [value] if field == "threshold" else [1.0]
    global_fraction = value if field == "global" else 0.0
    layer_fractions = [value] if field == "layer" else None
    scale = value if field == "scale" else 1.0
    if np.finfo(np.longdouble).max > np.finfo(np.float64).max:
        with pytest.raises(ValueError, match="finite"):
            ConvertedSNN(
                [[[1.0]]],
                [None],
                thresholds,
                T=1,
                initial_membrane_fraction=global_fraction,
                layer_membrane_fractions=layer_fractions,
                output_scale=scale,
            )
    else:
        ConvertedSNN(
            [[[1.0]]],
            [None],
            thresholds,
            T=1,
            initial_membrane_fraction=global_fraction,
            layer_membrane_fractions=layer_fractions,
            output_scale=scale,
        )
