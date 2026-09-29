# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ANN-to-SNN conversion engine

"""Convert actual source forward invocations to a normalized dense IF network.

The source is captured on an independent inference copy. Repeated module calls
remain distinct; unused registrations do not define topology. Consecutive affine
maps are composed without adding an IF nonlinearity. ReLU scales come from each
actual module/function/method invocation, or defaults (unit ReLU, six for ReLU6).
Each QCFS invocation keeps its learned theta and half-threshold membrane shift;
ReLU stages start from rest, including in mixed networks. The final affine-only
readout integrates signed current.

Source dtype rounding, finite-timestep quantization and spike timing can produce
conversion loss. Matching QCFS budgets does not guarantee lossless conversion;
source/target loss must be measured using the declared input encoding.

References
----------
Diehl et al. 2015 — "Fast-classifying, high-accuracy spiking deep networks
through weight and threshold balancing".
Bu et al. 2022 — "Optimal ANN-SNN Conversion for High-accuracy and
Ultra-low-latency Spiking Neural Networks" (ICLR).
"""

from __future__ import annotations

from .converted_snn import ConvertedSNN as ConvertedSNN
from .if_parameters import FloatArray, OutputMode
from .if_resources import DEFAULT_WORKING_BYTES

try:
    import torch
    import torch.nn as nn

    from .source_calibration import calibrate_source_nodes
    from .source_graph import SourcePlan, compile_source_graph
    from .qcfs import QCFSActivation

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _extract_layers(
    model: nn.Module, plan: SourcePlan | None = None
) -> list[tuple[FloatArray, FloatArray | None]]:
    """Extract owned affine coefficients in actual forward invocation order."""
    source = compile_source_graph(model, DEFAULT_WORKING_BYTES) if plan is None else plan
    return [(layer.weight, layer.bias) for layer in source.layers]


def _extract_qcfs_layers(
    model: nn.Module, plan: SourcePlan | None = None
) -> list[tuple[float, int]]:
    """Collect learned theta and T for every executed QCFS invocation."""
    source = compile_source_graph(model, DEFAULT_WORKING_BYTES) if plan is None else plan
    return [
        (layer.activation.theta, layer.activation.steps)
        for layer in source.layers
        if layer.activation.kind == "qcfs" and layer.activation.steps is not None
    ]


def replace_relu_with_qcfs(
    model: object,
    T: int = 8,
    theta: float = 1.0,
    learn_theta: bool = True,
) -> nn.Module:
    """Swap every ReLU/ReLU6 in a model for a QCFS activation, in place.

    This prepares a trained or fresh ANN for conversion-aware fine-tuning:
    after substitution the network is retrained for a few epochs so the QCFS
    thresholds settle. Conversion loss is then measured against the source ANN;
    QCFS does not guarantee lossless conversion for arbitrary spike timing.

    Parameters
    ----------
    model : nn.Module
        Model whose registered ReLU/ReLU6 children are replaced in place.
        All aliases of one original activation retain one shared QCFS module
        and learned threshold. The container and activation training modes
        are preserved; the root module itself is returned unchanged.
    T : int
        Quantisation step budget for each inserted QCFS layer.
    theta : float
        Initial firing threshold for each inserted QCFS layer.
    learn_theta : bool
        Whether each inserted threshold is a trainable parameter (the QCFS
        fine-tuning default).

    Returns
    -------
    nn.Module
        The same ``model`` instance, returned for chaining.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for ANN-to-SNN conversion")

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a PyTorch Module")

    replacements: dict[int, QCFSActivation] = {}
    # Snapshot unique parents; named_children omits duplicate registrations.
    for parent in tuple(model.modules()):
        for name, child in tuple(parent._modules.items()):
            if isinstance(child, (nn.ReLU, nn.ReLU6)):
                identity = id(child)
                if identity not in replacements:
                    replacements[identity] = QCFSActivation(
                        T=T, theta=theta, learn_theta=learn_theta
                    ).train(child.training)
                setattr(parent, name, replacements[identity])
    return model


def convert(
    model: object,
    calibration_data: object = None,
    T: int | None = None,
    percentile: float = 99.9,
    *,
    max_working_bytes: int = DEFAULT_WORKING_BYTES,
) -> ConvertedSNN:
    """Convert a trained PyTorch ANN to a rate-coded SNN.

    Activations are associated with their actual preceding source operations.
    ReLU and QCFS stages retain separate calibration scales and preloads in mixed
    networks. Unsupported dense-target operators fail compatibility admission.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model representable by a single-input dense forward path
        with Linear, ReLU/ReLU6 and QCFS operations. Actual invocations, including
        shared modules and functional ReLUs, determine the exported topology.
    calibration_data : Tensor, optional
        Sample source-format input for every ReLU invocation, including functional
        activations in mixed ReLU/QCFS networks. None uses unit ReLU scales;
        QCFS invocations always retain their learned thresholds.
    T : int, optional
        Number of simulation timesteps (higher = more accurate, slower). If
        None, the QCFS route adopts the layers' trained step budget and the
        ReLU route defaults to 16.
    percentile : float
        Activation percentile for threshold normalization on the ReLU route.

    max_working_bytes : int
        Numeric storage budget for source copying and exported snapshots.

    Returns
    -------
    ConvertedSNN
        Converted spiking network ready to run.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for ANN-to-SNN conversion")

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a PyTorch Module")
    plan = compile_source_graph(model, max_working_bytes)
    layers = _extract_layers(model, plan)
    qcfs_layers = _extract_qcfs_layers(model, plan)
    calibrated: dict[str, float] = {}
    has_relu = any(layer.activation.kind == "relu" for layer in plan.layers)
    if has_relu and calibration_data is not None:
        if not isinstance(calibration_data, torch.Tensor):
            raise TypeError("calibration_data must be a PyTorch Tensor")
        calibrated = calibrate_source_nodes(plan, calibration_data, percentile)
    thresholds = [
        layer.activation.theta
        if layer.activation.kind == "qcfs"
        else calibrated.get(layer.activation.node_name, layer.activation.theta)
        for layer in plan.layers
    ]
    if T is None:
        budgets = {budget for _, budget in qcfs_layers}
        if len(budgets) > 1:
            raise ValueError("mixed QCFS timestep budgets require an explicit T")
        T = qcfs_layers[0][1] if qcfs_layers else 16
    normalized_weights: list[FloatArray] = []
    normalized_biases: list[FloatArray | None] = []
    previous_scale = 1.0
    for (weight, bias), threshold in zip(layers, thresholds):
        normalized_weights.append(weight * previous_scale / threshold)
        normalized_biases.append(None if bias is None else bias / threshold)
        previous_scale = threshold
    output_mode: OutputMode = "linear" if plan.layers[-1].activation.kind == "linear" else "spikes"
    return ConvertedSNN(
        weights=normalized_weights,
        biases=normalized_biases,
        thresholds=[1.0] * len(layers),
        T=T,
        initial_membrane_fraction=0.5 if qcfs_layers else 0.0,
        output_scale=thresholds[-1],
        output_mode=output_mode,
        layer_membrane_fractions=[
            0.5 if layer.activation.kind == "qcfs" else 0.0 for layer in plan.layers
        ],
        max_working_bytes=max_working_bytes,
    )
