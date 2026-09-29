# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public converted network encoders, readouts and decoded rates

"""Exercise actual encoders, source-unit readouts and frame/state replay wiring."""

from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest
import torch

from sc_neurocore.conversion import ConvertedSNN, QCFSActivation, convert
from sc_neurocore.conversion.if_parameters import OutputMode


@pytest.mark.parametrize("mode", ["spikes", "linear"])
@pytest.mark.parametrize("encoding", ["poisson", "constant"])
def test_run_chunking_matches_complete_explicit_frames(
    mode: OutputMode,
    encoding: Literal["poisson", "constant"],
) -> None:
    """Keep the legacy MT19937 uniform order and all carried states across blocks."""
    rng = np.random.default_rng(97)
    model = ConvertedSNN(
        [rng.normal(size=(5, 3)), rng.normal(size=(2, 5))],
        [np.array([0.1, -0.1, 0.2, 0.3, -0.2]), None],
        [1.0, 1.0],
        T=130,
        output_mode=mode,
    )
    inputs = np.array([[0.25, 0.5, 0.75], [0.125, 0.625, 0.875]])
    frames = (
        (np.random.RandomState(42).random((130, 2, 3)) < inputs).astype(np.float64)
        if encoding == "poisson"
        else np.broadcast_to(inputs, (130, 2, 3))
    )
    expected = model.replay(frames, binary_inputs=encoding == "poisson")
    np.testing.assert_array_equal(model.run(inputs, input_mode=encoding), expected.output)


def test_source_unit_rates_and_signed_linear_readout_match_actual_ann() -> None:
    """Preserve a QCFS hidden activation and both positive and negative ANN logits."""
    first = torch.nn.Linear(2, 2).double()
    last = torch.nn.Linear(2, 2).double()
    with torch.no_grad():
        first.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64))
        assert first.bias is not None and last.bias is not None
        first.bias.zero_()
        last.weight.copy_(torch.tensor([[1.0, -2.0], [-2.0, 1.0]], dtype=torch.float64))
        last.bias.copy_(torch.tensor([0.125, -0.25], dtype=torch.float64))
    ann = torch.nn.Sequential(first, QCFSActivation(T=16, theta=2.0).double(), last)
    inputs = np.array([[0.25, 0.75], [0.75, 0.25]])
    reference = ann(torch.from_numpy(inputs)).detach().numpy()
    snn = convert(ann)
    assert snn.output_mode == "linear"
    np.testing.assert_array_equal(snn.rates(inputs, input_mode="constant"), reference)
    assert bool((reference < 0).any())


def test_spiking_output_retains_source_qcfs_scale() -> None:
    """Decode a nonunit QCFS threshold without caller-supplied scaling metadata."""
    linear = torch.nn.Linear(1, 1).double()
    with torch.no_grad():
        linear.weight.zero_()
        assert linear.bias is not None
        linear.bias.fill_(0.5)
    ann = torch.nn.Sequential(linear, QCFSActivation(T=16, theta=4.0).double())
    snn = convert(ann)
    assert snn.output_scale == 4.0 and snn.output_mode == "spikes"
    np.testing.assert_array_equal(snn.rates([0.0]), [0.5])


def test_single_vector_classification_and_empty_batch() -> None:
    """Preserve shape contracts and choose the first index when responses tie."""
    model = ConvertedSNN([[[1.0], [1.0]]], [None], [1.0], T=2)
    assert model.run([1.0]).shape == (2,)
    assert int(model.classify([1.0])) == 0
    np.testing.assert_array_equal(model.classify(np.ones((3, 1))), [0, 0, 0])
    assert model.run(np.empty((0, 1))).shape == (0, 2)


@pytest.mark.parametrize(
    "inputs",
    [
        np.array(0.5),
        np.zeros((1, 1, 1)),
        np.ones((1, 2)),
        np.array([-0.1]),
        np.array([1.1]),
        np.array([float("nan")]),
        np.array([1.0 + 1.0j]),
    ],
)
def test_invalid_input_refused(inputs: npt.ArrayLike) -> None:
    """Enforce declared real finite probability/current geometry at the public encoder."""
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=2)
    with pytest.raises(ValueError):
        model.run(inputs)


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_invalid_seed_refused(seed: int) -> None:
    """Require the declared unsigned 32-bit MT19937 seed representation."""
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=1)
    with pytest.raises(ValueError, match="seed"):
        model.run([0.5], seed=seed)


@pytest.mark.parametrize("encoding", ["unknown", "Poisson"])
def test_invalid_encoding_refused(encoding: Literal["poisson", "constant"]) -> None:
    """Reject silent changes to the named input event/current protocol."""
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=1)
    with pytest.raises(ValueError, match="input mode"):
        model.run([0.5], input_mode=encoding)


def test_mutated_budget_and_scale_revalidated() -> None:
    """Public field edits cannot bypass count precision or decoding domain checks."""
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=1)
    model.T = 0
    with pytest.raises(ValueError, match="positive integer"):
        model.run([0.5])
    model.T = 1
    model.output_scale = float("nan")
    with pytest.raises(ValueError, match="output scale"):
        model.rates([0.5])
