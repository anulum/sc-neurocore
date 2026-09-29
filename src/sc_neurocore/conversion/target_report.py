# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Target fixed-point compatibility and calibration of a converted network

"""Fit a converted network into one target's fixed-point format and measure the cost.

Integrate-and-fire dynamics with subtractive reset are invariant under scaling
one layer's weights, bias, threshold and membrane by the same positive factor:
the same neurons spike at the same steps. Each layer is therefore given the
finest power-of-two scale at which its coefficients, its threshold and the
membrane peak measured on the calibration samples fit the target's
representable range, and its coefficients are rounded (half to even) onto the
target's grid at that scale.

The network with rounded coefficients is then replayed on the same samples and
compared with the unrounded one. A layer driven by spikes accumulates exact
multiples of the grid step, so its float64 replay equals integer fixed-point
accumulation while the width is at most 53 bits and the registers do not
overflow; the replay counts every step at which a measured membrane leaves the
range. A layer driven by analog input multiplies it in float64, which the target
would round; that layer is not emulated and says so. Nothing here runs on, or
estimates anything about, the target's latency or energy.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from .converted_snn import ConvertedSNN
from .if_dispatch import ReplayBackend, resolve_replay_backend
from .if_resources import DEFAULT_WORKING_BYTES
from .loss_report import converted_sha256, data_sha256

if TYPE_CHECKING:
    from sc_neurocore.compiler.platforms.registry import HardwareProfile

TARGET_REPORT_SCHEMA_VERSION = "sc-neurocore.conversion-target-report.v1"
_CHUNK = 64
_EXACT_WIDTH = 53


@dataclass(frozen=True)
class LayerCalibration:
    """How one layer was fitted into the target format, and what that cost.

    Attributes
    ----------
    index : int
        Zero-based layer position.
    drive : {'analog', 'spikes'}
        What the layer integrates; only a spike-driven layer's replay is exact.
    readout : {'spiking', 'linear'}
        Whether the layer fires or integrates its current as the readout.
    scale_exponent : int
        ``e`` of the scale ``2**e`` applied to the layer's normalised values.
    threshold_code : int
        The threshold as a stored integer; zero for a linear readout.
    measured_peak : float
        Largest pre-reset membrane magnitude on the calibration samples, in
        normalised threshold units.
    headroom_bits : float or None
        ``log2`` of the representable maximum over the scaled peak; ``None``
        when the membrane never moved.
    weights : int
        Number of weights.
    zeroed_weights : int
        Non-zero weights that round to zero.
    weight_max_abs_error, weight_rms_error : float
        Rounding error of the weights in normalised units.
    bias_max_abs_error : float
        Rounding error of the bias; zero without a bias.
    preload_exact : bool
        Whether the initial membrane preload lies on the grid.
    overflow_steps : int
        Sample-steps at which the rounded network's membrane left the range.
    """

    index: int
    drive: Literal["analog", "spikes"]
    readout: Literal["spiking", "linear"]
    scale_exponent: int
    threshold_code: int
    measured_peak: float
    headroom_bits: float | None
    weights: int
    zeroed_weights: int
    weight_max_abs_error: float
    weight_rms_error: float
    bias_max_abs_error: float
    preload_exact: bool
    overflow_steps: int


@dataclass(frozen=True)
class TargetReport:
    """A converted network fitted into one target format and measured there.

    Attributes
    ----------
    schema_version : str
        ``sc-neurocore.conversion-target-report.v1``.
    profile : dict
        The target profile's identity and numeric format.
    compatible : bool
        Every layer fits with a threshold of at least one grid step and no
        measured overflow.
    refusals : list of str
        Why the network is not compatible; empty when it is.
    layers : list of LayerCalibration
        One entry per layer.
    samples, timesteps : int
        Calibration samples and the network's timestep budget.
    input_mode : str
        ``constant``: every sample drives the first layer as a current.
    backend : str
        Replay runtime both networks ran on.
    agreement : float
        Fraction of samples on which both networks predict the same class.
    output_max_abs_difference : float
        Largest decoded-output difference between the two networks.
    float_accuracy, quantized_accuracy, accuracy_drop : float or None
        Present when labels were given.
    exact_accumulation : bool
        Whether spike-driven layers' replay equals integer accumulation: the
        format is at most 53 bits wide and their preloads lie on the grid.
    converted_sha256, data_sha256 : str
        Digests of the unrounded network and of the calibration samples.
    arithmetic : str
        What the measurement emulates and what it does not.
    """

    schema_version: str
    profile: dict[str, Any]
    compatible: bool
    refusals: list[str]
    layers: list[LayerCalibration]
    samples: int
    timesteps: int
    input_mode: str
    backend: str
    agreement: float
    output_max_abs_difference: float
    float_accuracy: float | None
    quantized_accuracy: float | None
    accuracy_drop: float | None
    exact_accumulation: bool
    converted_sha256: str
    data_sha256: str
    arithmetic: str

    def to_public_dict(self) -> dict[str, Any]:
        """Return the report as JSON-ready fields.

        Returns
        -------
        dict
            Every attribute under its own name, layers as objects.
        """
        return asdict(self)


def _profile_identity(profile: HardwareProfile) -> dict[str, Any]:
    """Return the profile fields the report depends on."""
    return {
        "name": profile.name,
        "vendor": profile.vendor,
        "family": profile.family,
        "platform_class": profile.platform_class,
        "q_format": profile.q_format_label,
        "data_width": profile.data_width,
        "fraction": profile.fraction,
        "signed": profile.signed,
        "overflow": profile.overflow,
        "rounding": profile.rounding,
    }


def _code_bounds(profile: HardwareProfile) -> tuple[int, int]:
    """Return the smallest and largest stored integer."""
    if profile.signed:
        return -(1 << (profile.data_width - 1)), (1 << (profile.data_width - 1)) - 1
    return 0, (1 << profile.data_width) - 1


def _chunks(steps: int) -> Iterator[int]:
    """Yield replay chunk lengths covering ``steps``."""
    for start in range(0, steps, _CHUNK):
        yield min(_CHUNK, steps - start)


def _on_grid(
    values: npt.NDArray[np.float64], scale: float, grid: float, low: int, high: int
) -> npt.NDArray[np.float64]:
    """Round scaled values half to even onto the stored-integer grid and scale back."""
    codes = np.clip(np.rint(values * scale / grid), low, high)
    return np.asarray(codes * grid / scale, dtype=np.float64)


def _replay_with_peaks(
    snn: ConvertedSNN,
    samples: npt.NDArray[np.float64],
    batch_size: int,
    backend: ReplayBackend,
    max_working_bytes: int,
    limits: list[tuple[float, float]] | None,
) -> tuple[npt.NDArray[np.float64], list[float], list[int]]:
    """Replay constant-current samples, returning outputs, peaks and range exits.

    Peaks are pre-reset membrane magnitudes: the post-step state plus the
    threshold subtracted by a spike. ``limits`` are per-layer ``(lowest,
    highest)`` membranes; a sample-step with any neuron outside them counts once.
    """
    layers = snn.n_layers
    spiking = layers - int(snn.output_mode == "linear")
    peaks = [0.0] * layers
    exits = [0] * layers
    outputs = np.zeros((len(samples), snn.weights[-1].shape[0]), dtype=np.float64)
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        state: Any = None
        for length in _chunks(snn.T):
            frames = np.broadcast_to(batch, (length, *batch.shape))
            result = snn.replay(
                frames,
                initial_state=state,
                trace=True,
                binary_inputs=False,
                max_working_bytes=max_working_bytes,
                backend=backend,
            )
            state = result.final_state
            for index in range(layers):
                membrane = result.state_trace[index]
                if index < spiking:
                    membrane = membrane + result.spike_trace[index] * snn.thresholds[index]
                peaks[index] = max(peaks[index], float(np.abs(membrane).max(initial=0.0)))
                if limits is not None:
                    lowest, highest = limits[index]
                    outside = (membrane < lowest) | (membrane > highest)
                    exits[index] += int(np.count_nonzero(outside.any(axis=2)))
            if snn.output_mode == "linear":
                outputs[start : start + len(batch)] = result.output
            else:
                outputs[start : start + len(batch)] += result.output
    return outputs, peaks, exits


def calibrate_for_target(
    snn: ConvertedSNN,
    profile: HardwareProfile,
    inputs: npt.ArrayLike,
    labels: npt.ArrayLike | None = None,
    *,
    batch_size: int = 256,
    max_working_bytes: int = DEFAULT_WORKING_BYTES,
    backend: ReplayBackend = "auto",
) -> TargetReport:
    """Fit ``snn`` into ``profile``'s fixed-point format and measure the rounded network.

    Parameters
    ----------
    snn : ConvertedSNN
        Converted network in normalised threshold units.
    profile : HardwareProfile
        Target format from :mod:`sc_neurocore.compiler.platforms`.
    inputs : array_like
        ``(samples, input_neurons)`` calibration values in ``[0, 1]``, driven as
        constant currents for the network's timestep budget.
    labels : array_like, optional
        ``(samples,)`` integer classes; accuracies are reported when given.
    batch_size : int
        Positive samples per replay.
    max_working_bytes : int
        Numeric buffer budget of each replay chunk.
    backend : {'auto', 'numpy', 'rust', 'go', 'mojo', 'julia'}
        Replay runtime, resolved once and named in the report.

    Returns
    -------
    TargetReport
        Per-layer scales and rounding costs, measured ranges, both networks'
        agreement and, with labels, accuracies.

    Raises
    ------
    ValueError
        Empty or out-of-range inputs, invalid labels or batch size.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    samples = np.array(inputs, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("inputs must be a non-empty (samples, input_neurons) array")
    if not np.all(np.isfinite(samples)) or samples.min() < 0 or samples.max() > 1:
        raise ValueError("inputs must be finite values in [0, 1]")
    targets = None
    if labels is not None:
        targets = np.array(labels)
        if targets.dtype.kind not in "iu" or targets.shape != (samples.shape[0],):
            raise ValueError("labels must be one integer class per sample")
    executed = resolve_replay_backend(backend)
    float_out, peaks, _ = _replay_with_peaks(
        snn, samples, batch_size, executed, max_working_bytes, None
    )
    low, high = _code_bounds(profile)
    grid = 2.0**-profile.fraction
    top = high * grid
    fractions = snn.layer_membrane_fractions or [snn.initial_membrane_fraction] * snn.n_layers
    linear_index = snn.n_layers - 1 if snn.output_mode == "linear" else -1
    refusals: list[str] = []
    layers: list[LayerCalibration] = []
    weights: list[npt.NDArray[np.float64]] = []
    biases: list[npt.NDArray[np.float64] | None] = []
    thresholds: list[float] = []
    limits: list[tuple[float, float]] = []
    for index in range(snn.n_layers):
        weight = np.asarray(snn.weights[index], dtype=np.float64)
        bias = snn.biases[index]
        threshold = float(snn.thresholds[index])
        linear = index == linear_index
        extent = max(
            float(np.abs(weight).max(initial=0.0)),
            0.0 if bias is None else float(np.abs(bias).max(initial=0.0)),
            0.0 if linear else threshold,
            peaks[index],
        )
        exponent = math.floor(math.log2(top / extent)) if extent > 0 and top > 0 else 0
        scale = 2.0**exponent
        if not profile.signed and (
            weight.min(initial=0.0) < 0 or (bias is not None and bias.min() < 0)
        ):
            refusals.append(f"layer {index}: negative coefficients in an unsigned format")
        threshold_code = 0 if linear else int(np.rint(threshold * scale / grid))
        if not linear and threshold_code < 1:
            refusals.append(f"layer {index}: its dynamic range leaves the threshold below one step")
        weight_q = _on_grid(weight, scale, grid, low, high)
        error = np.abs(weight_q - weight)
        bias_q = None
        bias_error = 0.0
        if bias is not None:
            bias_q = _on_grid(np.asarray(bias, dtype=np.float64), scale, grid, low, high)
            bias_error = float(np.abs(bias_q - bias).max(initial=0.0))
        weights.append(weight_q)
        biases.append(bias_q)
        stored = threshold if linear else max(threshold_code, 1) * grid / scale
        thresholds.append(stored)
        stored_preload = 0.0 if linear else float(fractions[index]) * stored
        limits.append((low * grid / scale, top / scale))
        layers.append(
            LayerCalibration(
                index=index,
                drive="analog" if index == 0 else "spikes",
                readout="linear" if linear else "spiking",
                scale_exponent=exponent,
                threshold_code=threshold_code,
                measured_peak=peaks[index],
                headroom_bits=math.log2(top / (peaks[index] * scale)) if peaks[index] > 0 else None,
                weights=int(weight.size),
                zeroed_weights=int(np.count_nonzero((weight != 0) & (weight_q == 0))),
                weight_max_abs_error=float(error.max(initial=0.0)),
                weight_rms_error=float(np.sqrt(np.mean(error**2))) if error.size else 0.0,
                bias_max_abs_error=bias_error,
                preload_exact=bool(
                    np.rint(stored_preload * scale / grid) * grid / scale == stored_preload
                ),
                overflow_steps=0,
            )
        )
    quantized = ConvertedSNN(
        weights,
        biases,
        thresholds,
        snn.T,
        snn.initial_membrane_fraction,
        snn.output_scale,
        snn.output_mode,
        max_working_bytes=max_working_bytes,
        layer_membrane_fractions=snn.layer_membrane_fractions,
    )
    quant_out, _, exits = _replay_with_peaks(
        quantized, samples, batch_size, executed, max_working_bytes, limits
    )
    for index, count in enumerate(exits):
        layers[index] = LayerCalibration(**{**asdict(layers[index]), "overflow_steps": count})
        if count:
            refusals.append(f"layer {index}: membrane left the range at {count} sample-steps")
    float_classes = float_out.argmax(axis=1)
    quant_classes = quant_out.argmax(axis=1)
    decode = snn.output_scale / snn.T
    float_accuracy = quantized_accuracy = drop = None
    data_labels = np.zeros(0, dtype=np.int64) if targets is None else targets.astype(np.int64)
    if targets is not None:
        float_accuracy = float(np.mean(float_classes == targets))
        quantized_accuracy = float(np.mean(quant_classes == targets))
        drop = float_accuracy - quantized_accuracy
    return TargetReport(
        schema_version=TARGET_REPORT_SCHEMA_VERSION,
        profile=_profile_identity(profile),
        compatible=not refusals,
        refusals=refusals,
        layers=layers,
        samples=int(samples.shape[0]),
        timesteps=snn.T,
        input_mode="constant",
        backend=executed,
        agreement=float(np.mean(float_classes == quant_classes)),
        output_max_abs_difference=float(np.abs(quant_out - float_out).max() * decode),
        float_accuracy=float_accuracy,
        quantized_accuracy=quantized_accuracy,
        accuracy_drop=drop,
        exact_accumulation=profile.data_width <= _EXACT_WIDTH
        and all(layer.preload_exact for layer in layers[1:]),
        converted_sha256=converted_sha256(snn),
        data_sha256=data_sha256(samples, data_labels),
        arithmetic=(
            "Coefficients rounded half to even onto the target grid at a per-layer "
            "power-of-two scale; replayed in float64. Spike-driven layers equal integer "
            "accumulation when exact_accumulation is true; the analog-driven first layer's "
            "products are not rounded as the target would round them. Target overflow and "
            "rounding modes are recorded, not emulated. Only the numeric format is checked: "
            "neuron counts, fan-in, core mapping and routing are not, and latency and energy "
            "are not measured."
        ),
    )


__all__ = [
    "TARGET_REPORT_SCHEMA_VERSION",
    "LayerCalibration",
    "TargetReport",
    "calibrate_for_target",
]
