# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Measured source-to-converted conversion loss

"""Measure what conversion cost on labelled data, bound to the exact models and inputs.

The source ANN and its converted SNN classify the same samples. The report
states both accuracies, their agreement, the decoded-rate error against the
source output, the declared input encoding and timestep budget, the replay
runtime that actually executed, and digests of the source parameters, the
converted coefficients and the evaluated data. Nothing is estimated: every
number comes from running both networks on the given samples.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from .converted_snn import ConvertedSNN
from .if_dispatch import ReplayBackend, resolve_replay_backend
from .if_resources import DEFAULT_WORKING_BYTES

LOSS_REPORT_SCHEMA_VERSION = "sc-neurocore.conversion-loss-report.v1"
_SEED_LIMIT = 2**32


@dataclass(frozen=True)
class ConversionLossReport:
    """Measured agreement between a source ANN and its converted SNN.

    Attributes
    ----------
    schema_version : str
        ``sc-neurocore.conversion-loss-report.v1``.
    samples : int
        Number of labelled samples both networks classified.
    classes : int
        Width of the output both networks produce.
    timesteps : int
        The converted network's timestep budget.
    input_mode : {'constant', 'poisson'}
        Declared input encoding of the converted network.
    seed : int
        Poisson seed of the first batch; batch ``k`` uses ``(seed + k) mod 2**32``.
    batch_size : int
        Samples per converted-network call.
    backend : {'numpy', 'rust', 'go', 'mojo', 'julia'}
        Replay runtime every converted batch executed on.
    source_accuracy : float
        Fraction of samples the source ANN labels correctly.
    converted_accuracy : float
        Fraction of samples the converted SNN labels correctly.
    accuracy_drop : float
        ``source_accuracy - converted_accuracy``; positive is a loss.
    agreement : float
        Fraction of samples on which both networks predict the same class.
    rate_mean_abs_error : float
        Mean absolute difference between decoded SNN rates and source outputs.
    rate_max_abs_error : float
        Largest such difference.
    source_sha256 : str
        Digest of every source parameter and buffer, by name, dtype and shape.
    converted_sha256 : str
        Digest of the converted coefficients and replay semantics.
    data_sha256 : str
        Digest of the evaluated inputs, their shape and their labels.
    numerical_profile : str
        Arithmetic profile of the converted replay.
    """

    schema_version: str
    samples: int
    classes: int
    timesteps: int
    input_mode: Literal["constant", "poisson"]
    seed: int
    batch_size: int
    backend: Literal["numpy", "rust", "go", "mojo", "julia"]
    source_accuracy: float
    converted_accuracy: float
    accuracy_drop: float
    agreement: float
    rate_mean_abs_error: float
    rate_max_abs_error: float
    source_sha256: str
    converted_sha256: str
    data_sha256: str
    numerical_profile: str = "dense-if-f64-sequential-v1"

    def to_public_dict(self) -> dict[str, Any]:
        """Return the report as JSON-ready fields.

        Returns
        -------
        dict
            Every attribute under its own name.
        """
        return asdict(self)


def _framed(digest: Any, header: dict[str, Any], payload: bytes) -> None:
    """Feed one length-framed header and payload so no two layouts collide."""
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _array_bytes(array: npt.NDArray[Any]) -> bytes:
    """Return C-ordered little-endian bytes of an array."""
    contiguous = np.ascontiguousarray(array)
    return contiguous.astype(contiguous.dtype.newbyteorder("<"), copy=False).tobytes()


def converted_sha256(snn: ConvertedSNN) -> str:
    """Digest a converted network's coefficients and replay semantics.

    Parameters
    ----------
    snn : ConvertedSNN
        Network whose current public coefficients are digested.

    Returns
    -------
    str
        Hex SHA-256 over weights, biases, thresholds, budget, preloads,
        output scale and output mode.
    """
    digest = hashlib.sha256()
    _framed(
        digest,
        {
            "T": snn.T,
            "initial_membrane_fraction": float(snn.initial_membrane_fraction),
            "layer_membrane_fractions": snn.layer_membrane_fractions,
            "layers": snn.n_layers,
            "output_mode": snn.output_mode,
            "output_scale": float(snn.output_scale),
            "thresholds": [float(value) for value in snn.thresholds],
        },
        b"",
    )
    for index, (weight, bias) in enumerate(zip(snn.weights, snn.biases, strict=True)):
        matrix = np.asarray(weight, dtype=np.float64)
        _framed(digest, {"layer": index, "shape": list(matrix.shape)}, _array_bytes(matrix))
        if bias is None:
            _framed(digest, {"layer": index, "bias": None}, b"")
        else:
            vector = np.asarray(bias, dtype=np.float64)
            _framed(
                digest,
                {"layer": index, "bias": list(vector.shape)},
                _array_bytes(vector),
            )
    return digest.hexdigest()


def source_sha256(model: Any) -> str:
    """Digest every parameter and buffer of a PyTorch module.

    Parameters
    ----------
    model : torch.nn.Module
        Module whose state is digested in name order.

    Returns
    -------
    str
        Hex SHA-256 over each state entry's name, dtype, shape and C-ordered
        storage bytes. PyTorch runs only on little-endian hosts, so the bytes
        are little-endian for every dtype, including those NumPy lacks.
    """
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        _framed(
            digest,
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            raw.numpy().tobytes(),
        )
    return digest.hexdigest()


def data_sha256(inputs: npt.NDArray[np.float64], labels: npt.NDArray[np.int64]) -> str:
    """Digest evaluated inputs and labels.

    Parameters
    ----------
    inputs : ndarray
        Float64 samples in their source shape.
    labels : ndarray
        Int64 class labels, one per sample.

    Returns
    -------
    str
        Hex SHA-256 over the input shape and values followed by the labels.
    """
    digest = hashlib.sha256()
    _framed(digest, {"inputs": list(inputs.shape), "dtype": "float64"}, _array_bytes(inputs))
    _framed(digest, {"labels": list(labels.shape), "dtype": "int64"}, _array_bytes(labels))
    return digest.hexdigest()


def _batches(samples: int, batch_size: int) -> Iterator[slice]:
    """Yield consecutive sample slices of at most ``batch_size``."""
    for start in range(0, samples, batch_size):
        yield slice(start, min(start + batch_size, samples))


def _source_outputs(model: Any, inputs: npt.NDArray[np.float64], batch_size: int) -> Any:
    """Run the source in inference mode and restore every module's training flag."""
    import torch

    reference = next(iter(model.parameters()), None)
    dtype = torch.float32 if reference is None else reference.dtype
    device = torch.device("cpu") if reference is None else reference.device
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            outputs = [
                model(torch.as_tensor(inputs[part], dtype=dtype, device=device))
                .detach()
                .cpu()
                .to(torch.float64)
                .numpy()
                for part in _batches(len(inputs), batch_size)
            ]
    finally:
        for module, training in modes:
            module.training = training
    return np.concatenate(outputs, axis=0)


def measure_conversion_loss(
    model: Any,
    snn: ConvertedSNN,
    inputs: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    input_mode: Literal["constant", "poisson"] = "constant",
    seed: int = 42,
    batch_size: int = 256,
    max_working_bytes: int = DEFAULT_WORKING_BYTES,
    backend: ReplayBackend = "auto",
) -> ConversionLossReport:
    """Classify labelled samples with a source ANN and its converted SNN and compare.

    Parameters
    ----------
    model : torch.nn.Module
        Source network. It runs under ``no_grad`` in inference mode on the
        device and dtype of its first parameter; every module's training flag
        is restored afterwards.
    snn : ConvertedSNN
        The network converted from ``model``.
    inputs : array_like
        ``(samples, *source_shape)`` values in ``[0, 1]``. The converted
        network receives each sample flattened in C order.
    labels : array_like
        ``(samples,)`` integer classes in ``[0, classes)``.
    input_mode : {'constant', 'poisson'}
        Declared encoding for the converted network.
    seed : int
        Unsigned 32-bit Poisson seed of the first batch.
    batch_size : int
        Positive number of samples per call of either network.
    max_working_bytes : int
        Numeric buffer budget of each converted-network call.
    backend : {'auto', 'numpy', 'rust', 'go', 'mojo', 'julia'}
        Replay runtime. Auto is resolved once, and every batch runs on the
        runtime the report names.

    Returns
    -------
    ConversionLossReport
        Accuracies, agreement, rate error, executed runtime and digests.

    Raises
    ------
    ValueError
        Empty, non-finite or mis-shaped inputs, invalid labels, an invalid
        batch size or seed, or outputs of different widths.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(seed) is not int or not 0 <= seed < _SEED_LIMIT:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if input_mode not in ("constant", "poisson"):
        raise ValueError("input_mode must be 'constant' or 'poisson'")
    samples = np.array(inputs, dtype=np.float64)
    if samples.ndim < 2 or samples.shape[0] == 0:
        raise ValueError("inputs must hold at least one sample with its source shape")
    if not np.all(np.isfinite(samples)):
        raise ValueError("inputs must be finite")
    targets = np.array(labels)
    if targets.dtype.kind not in "iu" or targets.shape != (samples.shape[0],):
        raise ValueError("labels must be one integer class per sample")
    targets = targets.astype(np.int64)
    executed = resolve_replay_backend(backend)
    source = _source_outputs(model, samples, batch_size)
    if source.ndim != 2 or source.shape[0] != samples.shape[0]:
        raise ValueError("the source must produce one output vector per sample")
    classes = int(source.shape[1])
    if targets.min() < 0 or targets.max() >= classes:
        raise ValueError("labels must lie in [0, classes)")
    flat = samples.reshape(samples.shape[0], -1)
    rates = np.empty_like(source)
    responses = np.empty_like(source)
    for index, part in enumerate(_batches(len(flat), batch_size)):
        response = snn.run(
            flat[part],
            input_mode=input_mode,
            seed=(seed + index) % _SEED_LIMIT,
            max_working_bytes=max_working_bytes,
            backend=executed,
        )
        if response.shape != source[part].shape:
            raise ValueError("the converted network's output width differs from the source")
        responses[part] = response
        rates[part] = response / snn.T * snn.output_scale
    source_predictions = np.argmax(source, axis=1)
    converted_predictions = np.argmax(responses, axis=1)
    source_accuracy = float(np.mean(source_predictions == targets))
    converted_accuracy = float(np.mean(converted_predictions == targets))
    error = np.abs(rates - source)
    return ConversionLossReport(
        schema_version=LOSS_REPORT_SCHEMA_VERSION,
        samples=int(samples.shape[0]),
        classes=classes,
        timesteps=snn.T,
        input_mode=input_mode,
        seed=seed,
        batch_size=batch_size,
        backend=executed,
        source_accuracy=source_accuracy,
        converted_accuracy=converted_accuracy,
        accuracy_drop=source_accuracy - converted_accuracy,
        agreement=float(np.mean(source_predictions == converted_predictions)),
        rate_mean_abs_error=float(np.mean(error)),
        rate_max_abs_error=float(np.max(error)),
        source_sha256=source_sha256(model),
        converted_sha256=converted_sha256(snn),
        data_sha256=data_sha256(samples, targets),
    )


__all__ = [
    "LOSS_REPORT_SCHEMA_VERSION",
    "ConversionLossReport",
    "converted_sha256",
    "data_sha256",
    "measure_conversion_loss",
    "source_sha256",
]
