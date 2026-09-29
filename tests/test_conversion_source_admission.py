# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual source graph compatibility and custody refusals

"""Exercise source operator, resource and tensor refusals through public convert."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sc_neurocore.conversion import QCFSActivation, convert


class SourceForms(nn.Module):
    """Execute genuine Torch activation and argument forms selected at construction."""

    def __init__(self, form: str) -> None:
        """Create a fixed identity source whose static mode determines its forward."""
        super().__init__()
        self.form = form
        self.linear = nn.Linear(2, 2, bias=False).double()
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(2, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use source function, method or keyword calls rather than test doubles."""
        output: torch.Tensor = self.linear(input=x) if self.form == "keyword" else self.linear(x)
        if self.form == "functional":
            return F.relu(output)
        if self.form == "relu6":
            return F.relu6(output)
        if self.form == "method":
            return output.relu()
        if self.form == "inplace":
            return output.relu_()
        return output


@pytest.mark.parametrize("form", ["functional", "relu6", "method", "inplace", "keyword"])
def test_actual_function_method_and_keyword_source_forms(form: str) -> None:
    """Calibrate actual source nodes and retain the correct final response type."""
    source = SourceForms(form)
    calibration = torch.tensor([[2.0, 1.0]], dtype=torch.float64)
    snn = convert(source, calibration_data=calibration, percentile=100)
    assert snn.output_mode == ("linear" if form == "keyword" else "spikes")
    assert snn.output_scale == (1.0 if form == "keyword" else 2.0)


def test_inference_identity_dropout_and_flatten_retain_signed_readout() -> None:
    """Actual inference no-ops do not turn a signed final affine map into IF."""
    linear = nn.Linear(2, 1, bias=False).double()
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[-1.0, 0.5]], dtype=torch.float64))
    source = nn.Sequential(nn.Flatten(), linear, nn.Identity(), nn.Dropout(0.9))
    snn = convert(source, T=8)
    inputs = np.array([[1.0, 0.0]])
    assert source.training and snn.output_mode == "linear"
    np.testing.assert_array_equal(snn.rates(inputs, input_mode="constant"), [[-1.0]])


@pytest.mark.parametrize("theta", [0.0, float("nan"), float("inf")])
def test_mutated_qcfs_threshold_refused(theta: float) -> None:
    """A learned threshold edited outside training cannot bypass source admission."""
    activation = QCFSActivation()
    with torch.no_grad():
        activation.theta.fill_(theta)
    with pytest.raises(ValueError, match="theta"):
        convert(nn.Sequential(nn.Linear(1, 1), activation))


@pytest.mark.parametrize("steps", [0, True])
def test_mutated_qcfs_grid_refused(steps: int) -> None:
    """Validate the trained lattice before selecting a simulation timestep budget."""
    activation = QCFSActivation()
    activation.T = steps
    with pytest.raises(ValueError, match="QCFS T"):
        convert(nn.Sequential(nn.Linear(1, 1), activation))


def test_complex_source_weights_are_not_silently_projected_to_real() -> None:
    """Reject a real Torch complex parameter before any lossy double conversion."""
    source = nn.Linear(2, 2, bias=False)
    source.weight = nn.Parameter(torch.eye(2, dtype=torch.complex128))
    with pytest.raises(ValueError, match="coefficients must be real"):
        convert(source)


def test_source_tensor_copy_admission_precedes_deepcopy() -> None:
    """Refuse independent source storage under a genuine too-small operator budget."""
    source = nn.Linear(2, 2, bias=False).double()
    with pytest.raises(MemoryError, match="source model copy"):
        convert(source, max_working_bytes=63)


def test_affine_fusion_admission_precedes_matrix_product() -> None:
    """A small factored source cannot authorize a much larger fused dense target."""
    source = nn.Sequential(
        nn.Linear(256, 1, bias=False).double(), nn.Linear(1, 256, bias=False).double()
    )
    with pytest.raises(MemoryError, match="fused affine"):
        convert(source, max_working_bytes=9000)


def test_adjacent_activations_identity_admission_precedes_allocation() -> None:
    """Two activations require an admitted identity operator between their stages."""
    source = nn.Sequential(nn.Linear(1, 128, bias=False).double(), nn.ReLU(), QCFSActivation())
    with pytest.raises(MemoryError, match="activation identity"):
        convert(source, max_working_bytes=9000)


def test_adjacent_activations_lower_to_distinct_real_stages() -> None:
    """Preserve both actual activations rather than overwriting their association."""
    source = nn.Sequential(
        nn.Linear(1, 1, bias=False).double(), nn.ReLU(), QCFSActivation(T=8).double()
    )
    snn = convert(source)
    assert snn.n_layers == 2 and snn.layer_membrane_fractions == [0.0, 0.5]
    np.testing.assert_array_equal(snn.weights[1], [[1.0]])


@pytest.mark.parametrize(
    "first_bias,last_bias", [(False, False), (False, True), (True, False), (True, True)]
)
def test_affine_fusion_preserves_each_actual_bias_case(first_bias: bool, last_bias: bool) -> None:
    """Compose signed weights and optional offsets without an invented activation."""
    first = nn.Linear(1, 1, bias=first_bias).double()
    last = nn.Linear(1, 1, bias=last_bias).double()
    with torch.no_grad():
        first.weight.fill_(-0.5)
        last.weight.fill_(0.5)
        if first.bias is not None:
            first.bias.fill_(0.25)
        if last.bias is not None:
            last.bias.fill_(-0.125)
    source = nn.Sequential(first, last)
    snn = convert(source, T=8)
    inputs = np.array([[1.0]])
    assert snn.n_layers == 1 and snn.output_mode == "linear"
    np.testing.assert_array_equal(
        snn.rates(inputs, input_mode="constant"), source(torch.from_numpy(inputs)).detach().numpy()
    )


@pytest.mark.parametrize(
    "source",
    [
        nn.Sequential(nn.Dropout()),
        nn.Sequential(nn.Flatten(0), nn.Linear(2, 2)),
        nn.Sequential(nn.Conv2d(1, 1, 1)),
    ],
)
def test_absent_or_unrepresentable_dense_operators_are_explicit(source: nn.Module) -> None:
    """Refuse unsupported target operations instead of extracting plausible weights."""
    with pytest.raises(ValueError, match="No Linear|dense.*operation"):
        convert(source)


def test_real_activation_hook_replacing_tensor_is_refused() -> None:
    """A legal Torch hook can produce an incompatible source activation result."""
    source = nn.Sequential(nn.Linear(1, 1), nn.ReLU())

    def scalar_output(
        module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> int:
        """Replace the actual source output with a scalar using Torch's hook API."""
        return 1

    handle = source[1].register_forward_hook(scalar_output)
    with pytest.raises(ValueError, match="activation must be a tensor"):
        convert(source, calibration_data=torch.ones((1, 1)))
    assert handle.id in source[1]._forward_hooks
    handle.remove()


def test_real_activation_hook_replacing_output_with_empty_tensor_is_refused() -> None:
    """Do not compute an activation percentile from a genuinely empty source result."""
    source = nn.Sequential(nn.Linear(1, 1), nn.ReLU())

    def empty_output(
        module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> torch.Tensor:
        """Replace the source output with a legal zero-row tensor."""
        return output[:0]

    handle = source[1].register_forward_hook(empty_output)
    with pytest.raises(ValueError, match="finite and nonempty"):
        convert(source, calibration_data=torch.ones((1, 1)))
    handle.remove()


def test_multiple_outputs_require_an_explicitly_supported_target() -> None:
    """A tuple output cannot silently drop a source result from the export."""

    class Multiple(nn.Module):
        """Return an actual pair of tensor outputs."""

        def __init__(self) -> None:
            """Create an actual weighted branch alongside the original input."""
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Expose both original and transformed source outputs."""
            output: torch.Tensor = self.linear(x)
            return output, x

    with pytest.raises(ValueError, match="single tensor output"):
        convert(Multiple())


def test_constant_input_path_cannot_be_mistaken_for_variable_input() -> None:
    """A constant-only data path needs a different target input contract."""

    class Constant(SourceForms):
        """Use a fixed tensor instead of the provided source input."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Execute the real linear layer on a constant tensor."""
            output: torch.Tensor = self.linear(torch.ones(2, dtype=torch.float64))
            return output

    with pytest.raises(ValueError, match="dense.*operation"):
        convert(Constant("keyword"))


def test_multiple_required_inputs_report_source_compatibility_failure() -> None:
    """The single-input target refuses a source requiring two input tensors."""

    class MultipleInputs(nn.Module):
        """Require a genuine second forward input."""

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Add the two source inputs."""
            return x + y

    with pytest.raises(ValueError, match="single-input tracing"):
        convert(MultipleInputs())


def test_bfloat16_source_activation_calibrates_without_lossy_numpy_cast() -> None:
    """Convert real Torch bfloat16 values to supported statistics storage."""
    linear = nn.Linear(1, 1, bias=False).bfloat16()
    with torch.no_grad():
        linear.weight.fill_(1.0)
    source = nn.Sequential(linear, nn.ReLU())
    snn = convert(
        source, calibration_data=torch.tensor([[2.0], [1.0]], dtype=torch.bfloat16), percentile=100
    )
    assert snn.output_scale == 2.0
    np.testing.assert_array_equal(snn.rates([1.0], input_mode="constant"), [1.0])


def test_virtual_huge_source_calibration_refused_before_validation_or_forward() -> None:
    """Inspect Tensor metadata before finite masks or inference outputs allocate."""
    source = nn.Sequential(nn.Linear(1, 1, bias=False).double(), nn.ReLU())
    calibration = torch.ones((1, 1), dtype=torch.float64).expand(2**30, 1)
    with pytest.raises(MemoryError, match="source calibration buffers"):
        convert(source, calibration_data=calibration)


def test_scalar_calibration_input_reports_actual_source_geometry_failure() -> None:
    """A finite scalar is still incompatible with a real dense source input."""
    source = nn.Sequential(nn.Linear(1, 1).double(), nn.ReLU())
    with pytest.raises(RuntimeError):
        convert(source, calibration_data=torch.tensor(1.0, dtype=torch.float64))


def test_invalid_leaf_keyword_arguments_are_not_discarded() -> None:
    """Symbolic leaf calls still require the actual supported operator signature."""

    class Invalid(SourceForms):
        """Pass an unsupported keyword to a real Torch leaf module."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Expose the invalid source call for target compatibility admission."""
            output: torch.Tensor = self.linear(x, unexpected=True)
            return output

    with pytest.raises(ValueError, match="dense.*arguments"):
        convert(Invalid("keyword"))


def test_relu_replacement_requires_a_real_torch_model() -> None:
    """The public preparation helper rejects a nonmodel in a real Torch environment."""
    from sc_neurocore.conversion import replace_relu_with_qcfs

    with pytest.raises(TypeError, match="PyTorch Module"):
        replace_relu_with_qcfs(object())
