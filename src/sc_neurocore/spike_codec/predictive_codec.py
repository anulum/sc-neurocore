# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Predictive spike codec: only transmit surprises

"""Encode binary spike rasters with deterministic XOR predictors and ISI residuals.

Each packet selects its predictor parameters. Encoding and decoding start from
the same model state and update it from observed or recovered samples. Lossless
packets reproduce the original raster. Quantised packets run the predictor on
complete timing bins and expose those bins on decode.
"""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass
from math import isfinite
from operator import index
from struct import Struct

import numpy as np

from ._predictive_kernels import (
    _LFSR_MASK as _LFSR_MASK,
    _RatePredictor as _RatePredictor,
    _lfsr16_step as _lfsr16_step,
    _predict_and_xor as _predict_and_xor,
    _predict_and_xor_context as _predict_and_xor_context,
    _predict_and_xor_lfsr as _predict_and_xor_lfsr,
    _xor_and_recover as _xor_and_recover,
    _xor_and_recover_context as _xor_and_recover_context,
    _xor_and_recover_lfsr as _xor_and_recover_lfsr,
)
from .codec import CompressionResult, SpikeCodec, _dimension

_EMA = Struct("!dd")
_LFSR = Struct("!HH")
_CONTEXT = Struct("!B")
_WORLD_MODEL = Struct("!BdH")
_WORLD_MODEL_THRESHOLD = Struct("!BdHd")
_QUANTISED_GRID = Struct("!IQ")

# Optional native predictor components.
_HAS_RUST = False
_rust_predict_ema: Any = None
_rust_predict_lfsr: Any = None
_rust_recover_ema: Any = None
_rust_recover_lfsr: Any = None

try:
    from sc_neurocore_engine import (
        py_predict_xor_ema,
        py_predict_xor_lfsr,
        py_recover_xor_ema,
        py_recover_xor_lfsr,
    )

    _rust_predict_ema = py_predict_xor_ema
    _rust_predict_lfsr = py_predict_xor_lfsr
    _rust_recover_ema = py_recover_xor_ema
    _rust_recover_lfsr = py_recover_xor_lfsr
    _HAS_RUST = True  # pragma: no cover
except (ImportError, AttributeError):  # pragma: no cover
    _HAS_RUST = False


try:
    from sc_neurocore_engine import py_predict_xor_context, py_recover_xor_context
except ImportError:
    py_predict_xor_context = None
    py_recover_xor_context = None


def _binary_raster(spikes: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Validate samples before narrowing their representation to int8."""
    raster = np.asarray(spikes)
    if raster.ndim != 2 or raster.dtype.kind not in "biuf":
        raise ValueError("Predictive codec needs a two-dimensional binary raster")
    _dimension(raster.shape[0])
    _dimension(raster.shape[1])
    if not np.logical_or(raster == 0, raster == 1).all():
        raise ValueError("Predictive raster samples must be 0 or 1")
    return raster.astype(np.int8, copy=False)


def _uint(value: int, minimum: int, maximum: int, name: str) -> int:
    """Validate one exact wire integer before doing any predictor work."""
    value = index(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _finite_model_parameters(alpha: float, threshold: float) -> None:
    """Refuse non-finite floating-point predictor state."""
    if not (isfinite(alpha) and isfinite(threshold)):
        raise ValueError("Predictor alpha and threshold must be finite")


def _read_predictor_header(data: bytes, layout: Struct) -> tuple[tuple[Any, ...], bytes]:
    """Return complete fields and the exact remaining residual packet."""
    end = 4 + layout.size
    if len(data) < end:
        raise ValueError("Truncated predictive header")
    return layout.unpack_from(data, 4), data[end:]


@dataclass
class PredictiveCompressionResult(CompressionResult):
    """Compression result with predictive coding metrics."""

    prediction_accuracy: float = 0.0
    error_sparsity: float = 0.0
    predictor_type: str = "ema"


class PredictiveSpikeCodec:
    """Predictive spike codec: compress prediction errors, not raw spikes.

    Four predictor modes:
        'ema' (default): float EMA rate tracking + threshold comparison.
        'lfsr': Q8.8 rate tracking with the 16-bit generator recurrence
                used by sc_bitstream_encoder.v.
        'context': Markov context predictor. Hashes last K spike states
                   per channel, predicts from accumulated statistics.
        'world_model': Learnable autoregressive predictor (LMS-trained).
                   Predicts spike[t] from spike[t-K:t] via linear model
                   with sigmoid activation. Learns cross-channel correlations.

    Compression pipeline:
        1. For each timestep t:
           a. predicted[t] = predictor.predict()
           b. error[t] = actual[t] XOR predicted[t]
           c. predictor.update(actual[t])
        2. ISI-compress the error matrix
        3. Pack with header (predictor params for decoder sync)

    Parameters
    ----------
    alpha : float
        EMA smoothing factor or world-model learning rate.
    threshold : float
        Spike prediction threshold in EMA and world-model modes.
    predictor : str
        'ema', 'lfsr', 'context', or 'world_model'.
    alpha_q8 : int
        Q8.8 smoothing factor for lfsr mode. 1 = 1/256 ≈ 0.004.
    seed : int
        Unsigned 16-bit seed for LFSR or world-model mode. LFSR requires a nonzero seed.
    context_bits : int
        Number of past time steps (default 8). Context accepts 0–63 and
        the world model accepts 1–255.
    base_mode : str
        'lossless' or 'lossy' for the underlying ISI codec.
    timing_precision : int
        Number of original rows in each complete bin in lossy mode.
        Prediction runs on the quantised grid. The final partial bin is dropped.
    backend : {"auto", "python", "rust"}
        Select a complete available codec path. Auto uses native components
        when available. Rust requires the matching predictor and ISI components.
        The world-model implementation uses NumPy for probability evaluation
        and can use Rust for its least-mean-squares state update.
    """

    HEADER_MAGIC = b"PSCX"  # Predictive Spike Codec XOR (EMA)
    HEADER_MAGIC_LFSR = b"PSCL"  # Predictive Spike Codec LFSR
    HEADER_MAGIC_CTX = b"PSCC"  # Predictive Spike Codec Context
    HEADER_MAGIC_WM = b"PSCW"  # Predictive Spike Codec World Model
    HEADER_MAGIC_WM_THRESHOLD = b"PSCT"
    HEADER_MAGIC_QUANTISED = b"PSCQ"

    def __init__(
        self,
        alpha: float = 0.005,
        threshold: float = 0.5,
        predictor: str = "ema",
        alpha_q8: int = 1,
        seed: int = 0xACE1,
        context_bits: int = 8,
        base_mode: str = "lossless",
        timing_precision: int = 1,
        *,
        backend: str = "auto",
    ) -> None:
        """Initialise predictor settings and the underlying raster codec."""
        self.alpha = alpha
        self.threshold = threshold
        self.predictor = predictor
        self.alpha_q8 = alpha_q8
        self.seed = seed
        self.context_bits = context_bits
        self._backend = backend
        # Context predictor benefits from Huffman backend (sparse scattered errors)
        entropy = "huffman" if predictor in ("context", "world_model") else "auto"
        self.base_codec = SpikeCodec(
            mode=base_mode,
            timing_precision=timing_precision,
            entropy=entropy,
            backend=backend,
        )

    def _use_native(self, available: bool, predictor: str) -> bool:
        """Select the requested backend for the packet's actual predictor."""
        if self._backend == "python":
            return False
        if self._backend == "rust" and not available:
            raise ValueError(
                f"backend 'rust' needs sc_neurocore_engine with {predictor} prediction"
            )
        return available

    def compress(self, spikes: np.ndarray[Any, Any]) -> tuple[bytes, PredictiveCompressionResult]:
        """Compress spike raster using predictive error coding.

        Parameters
        ----------
        spikes : ndarray of shape (T, N), binary (int8 or bool)

        Returns
        -------
        (compressed_bytes, PredictiveCompressionResult)
        """
        spikes = _binary_raster(spikes)
        rows, channels = spikes.shape
        precision = self.base_codec._precision(rows)
        residual_codec = self.base_codec
        prefix = b""
        raster = spikes
        if precision > 1:
            bins = rows // precision
            raster = spikes[: bins * precision].reshape(bins, precision, channels)
            raster = (raster > 0).any(axis=1).astype(np.int8)
            residual_codec = self._lossless_residual_codec()
            prefix = self.HEADER_MAGIC_QUANTISED + _QUANTISED_GRID.pack(rows, precision)

        header, errors, correct_predictions = self._predict_residual(raster)
        residual, _ = residual_codec.compress(errors)
        encoded = prefix + header + residual
        original_bits = rows * channels
        compressed_bits = len(encoded) * 8
        predicted_bits = max(raster.size, 1)
        return encoded, PredictiveCompressionResult(
            original_bits=original_bits,
            compressed_bits=compressed_bits,
            compression_ratio=original_bits / compressed_bits,
            n_spikes=int(np.sum(spikes)),
            n_neurons=channels,
            n_timesteps=rows,
            lossless=self.base_codec.mode == "lossless",
            prediction_accuracy=correct_predictions / predicted_bits,
            error_sparsity=1.0 - int(np.sum(errors)) / predicted_bits,
            predictor_type=self.predictor,
        )

    def _lossless_residual_codec(self) -> SpikeCodec:
        """Encode or decode the residual grid at its own full resolution."""
        return SpikeCodec(entropy=self.base_codec.entropy, backend=self.base_codec.backend)

    def _predict_residual(
        self, spikes: np.ndarray[Any, Any]
    ) -> tuple[bytes, np.ndarray[Any, Any], int]:
        """Create the predictor header and exact residuals for the input grid."""
        T, N = spikes.shape
        if self.predictor == "world_model":
            _finite_model_parameters(self.alpha, self.threshold)
            history_len = _uint(self.context_bits, 1, 255, "world-model history length")
            seed = _uint(self.seed, 0, 65535, "world-model seed")
            from . import _world_model as spike_predictor

            use_native = self._use_native(
                spike_predictor.py_update_spike_world_model is not None, "world-model"
            )
            errors, correct_predictions = spike_predictor.predict_and_xor_world_model(
                spikes,
                N,
                history_len=history_len,
                lr=self.alpha,
                threshold=self.threshold,
                seed=seed,
                use_native=use_native,
            )
            if self.threshold == 0.5:
                header = self.HEADER_MAGIC_WM + _WORLD_MODEL.pack(history_len, self.alpha, seed)
            else:
                header = self.HEADER_MAGIC_WM_THRESHOLD + _WORLD_MODEL_THRESHOLD.pack(
                    history_len, self.alpha, seed, self.threshold
                )
        elif self.predictor == "context":
            context_bits = _uint(self.context_bits, 0, 63, "context history length")
            if self._use_native(py_predict_xor_context is not None, "context"):
                assert py_predict_xor_context is not None
                flat = np.ascontiguousarray(spikes).ravel()
                err_flat, correct_predictions = py_predict_xor_context(flat, N, context_bits)
                errors = np.asarray(err_flat).reshape(T, N)
            else:
                errors, correct_predictions = _predict_and_xor_context(spikes, N, context_bits)
            header = self.HEADER_MAGIC_CTX + _CONTEXT.pack(context_bits)
        elif self.predictor == "lfsr":
            alpha_q8 = _uint(self.alpha_q8, 0, 65535, "LFSR alpha_q8")
            seed = _uint(self.seed, 1, 65535, "LFSR seed")
            if self._use_native(_HAS_RUST, "LFSR") and N:
                flat = np.ascontiguousarray(spikes).ravel()
                err_flat, correct_predictions = _rust_predict_lfsr(
                    flat,
                    N,
                    alpha_q8,
                    seed,
                )
                errors = np.asarray(err_flat).reshape(T, N)
            else:
                errors, correct_predictions = _predict_and_xor_lfsr(
                    spikes,
                    N,
                    alpha_q8,
                    seed,
                )
            header = self.HEADER_MAGIC_LFSR + _LFSR.pack(alpha_q8, seed)
        elif self.predictor == "ema":
            _finite_model_parameters(self.alpha, self.threshold)
            if self._use_native(_HAS_RUST, "EMA") and N:
                flat = np.ascontiguousarray(spikes).ravel()
                err_flat, correct_predictions = _rust_predict_ema(
                    flat,
                    N,
                    self.alpha,
                    self.threshold,
                )
                errors = np.asarray(err_flat).reshape(T, N)
            else:
                errors, correct_predictions = _predict_and_xor(
                    spikes,
                    N,
                    self.alpha,
                    self.threshold,
                )
            header = self.HEADER_MAGIC + _EMA.pack(self.alpha, self.threshold)
        else:
            raise ValueError(f"Invalid predictor: {self.predictor!r}")

        return header, errors, correct_predictions

    def decompress(self, data: bytes, T: int, N: int) -> np.ndarray[Any, Any]:
        """Decompress to spike raster.

        Runs identical predictor on decoder side. XOR(error, predicted)
        recovers original spikes. Predictor type auto-detected from header.

        Parameters
        ----------
        data : bytes
            Compressed data from compress().
        T, N : int
            Original dimensions. For a quantised packet, T can also be the
            number of complete bins.

        Returns
        -------
        ndarray of shape (T, N), int8
            Quantised events are at bin indices. In the original-sized
            layout the rows after the last bin contain zeros.
        """
        T, N = _dimension(T), _dimension(N)
        magic = data[:4]
        if magic == self.HEADER_MAGIC_QUANTISED:
            return self._decode_quantised(data, T, N)
        codec = (
            self._lossless_residual_codec()
            if magic == self.HEADER_MAGIC_WM_THRESHOLD
            else self.base_codec
        )
        return self._recover_residual(data, T, N, codec)

    def _decode_quantised(self, data: bytes, rows: int, channels: int) -> np.ndarray[Any, Any]:
        """Read the original shape and recover on the compact quantised grid."""
        end = 4 + _QUANTISED_GRID.size
        if len(data) < end:
            raise ValueError("Truncated quantised predictive header")
        original_rows, precision = _QUANTISED_GRID.unpack_from(data, 4)
        if not 1 < precision <= original_rows + 1:
            raise ValueError("Invalid predictive timing precision")
        bins = original_rows // precision
        if rows not in (original_rows, bins):
            raise ValueError("Predictive packet has incompatible row count")
        raster = self._recover_residual(data[end:], bins, channels, self._lossless_residual_codec())
        if rows == bins:
            return raster
        result = np.zeros((rows, channels), dtype=np.int8)
        result[:bins] = raster
        return result

    def _recover_residual(
        self, data: bytes, T: int, N: int, residual_codec: SpikeCodec
    ) -> np.ndarray[Any, Any]:
        """Recover the raster with the complete predictor configuration from the packet."""
        magic = data[:4]

        if magic in (self.HEADER_MAGIC_WM, self.HEADER_MAGIC_WM_THRESHOLD):
            if magic == self.HEADER_MAGIC_WM:
                (history_len, alpha, seed), error_data = _read_predictor_header(data, _WORLD_MODEL)
                threshold = 0.5
            else:
                (history_len, alpha, seed, threshold), error_data = _read_predictor_header(
                    data, _WORLD_MODEL_THRESHOLD
                )
            _uint(history_len, 1, 255, "world-model history length")
            _finite_model_parameters(alpha, threshold)
            errors = residual_codec.decompress(error_data, T, N)
            from . import _world_model as spike_predictor

            use_native = self._use_native(
                spike_predictor.py_update_spike_world_model is not None, "world-model"
            )
            return spike_predictor.xor_and_recover_world_model(
                errors,
                N,
                history_len=history_len,
                lr=alpha,
                threshold=threshold,
                seed=seed,
                use_native=use_native,
            )

        if magic == self.HEADER_MAGIC_CTX:
            (context_bits,), error_data = _read_predictor_header(data, _CONTEXT)
            _uint(context_bits, 0, 63, "context history length")
            errors = residual_codec.decompress(error_data, T, N)
            if self._use_native(py_recover_xor_context is not None, "context"):
                assert py_recover_xor_context is not None
                flat = np.ascontiguousarray(errors).ravel()
                return np.asarray(py_recover_xor_context(flat, N, context_bits)).reshape(T, N)
            return _xor_and_recover_context(errors, N, context_bits)

        if magic == self.HEADER_MAGIC_LFSR:
            (alpha_q8, seed), error_data = _read_predictor_header(data, _LFSR)
            errors = residual_codec.decompress(error_data, T, N)
            if self._use_native(_HAS_RUST, "LFSR") and N:
                flat = np.ascontiguousarray(errors).ravel()
                rec = np.asarray(_rust_recover_lfsr(flat, N, alpha_q8, seed))
                return rec.reshape(T, N)
            return _xor_and_recover_lfsr(errors, N, alpha_q8, seed)

        if magic == self.HEADER_MAGIC:
            (alpha, threshold), error_data = _read_predictor_header(data, _EMA)
            _finite_model_parameters(alpha, threshold)
            errors = residual_codec.decompress(error_data, T, N)
            if self._use_native(_HAS_RUST, "EMA") and N:
                flat = np.ascontiguousarray(errors).ravel()
                rec = np.asarray(_rust_recover_ema(flat, N, alpha, threshold))
                return rec.reshape(T, N)
            return _xor_and_recover(errors, N, alpha, threshold)

        raise ValueError(
            f"Invalid header magic: {magic!r}, expected {self.HEADER_MAGIC!r} or {self.HEADER_MAGIC_LFSR!r}"
        )
