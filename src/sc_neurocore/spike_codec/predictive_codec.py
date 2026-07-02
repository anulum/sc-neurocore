# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Predictive spike codec: only transmit surprises

"""Predictive spike compression: XOR-based prediction error coding.

Architecture:
    1. Maintain per-channel firing rate predictor (exponential moving average)
    2. At each timestep, predict spike pattern from learned rates
    3. XOR actual vs predicted → prediction error (surprise) bits
    4. ISI-compress only the error bits (sparser than raw spikes)
    5. Decoder runs identical predictor → lossless reconstruction

The predictor is deterministic given the same seed, so encoder and decoder
stay synchronized without transmitting predictor state.

Neuralink context: 1024+ channels at 20 kHz produce ~200 Mbps raw.
Typical cortical neurons fire at 0.5-5 Hz → >99.9% of bits are zeros.
ISI coding alone gives 50-200x. Predictive coding removes the remaining
structured correlations (bursts, oscillations, drift) for additional 2-5x
on top of ISI baseline.
"""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass

import numpy as np

from .codec import SpikeCodec, CompressionResult
from ..world_model.spike_predictor import predict_and_xor_world_model, xor_and_recover_world_model

# Rust backend (optional, ~100x faster for LFSR predictor)
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


@dataclass
class PredictiveCompressionResult(CompressionResult):
    """Compression result with predictive coding metrics."""

    prediction_accuracy: float = 0.0
    error_sparsity: float = 0.0
    predictor_type: str = "ema"


class _RatePredictor:
    """Per-channel EMA rate predictor (legacy, kept for reference)."""

    def __init__(self, n_channels: int, alpha: float = 0.005, threshold: float = 0.5) -> None:
        """Initialize per-channel rates and threshold parameters."""
        self.n_channels = n_channels
        self.alpha = alpha
        self.threshold = threshold
        self.rates = np.zeros(n_channels, dtype=np.float64)

    def predict(self) -> np.ndarray[Any, Any]:
        """Predict active channels from the current EMA rate estimates."""
        return (self.rates > self.threshold).astype(np.int8)

    def update(self, actual: np.ndarray[Any, Any]) -> None:
        """Update EMA rates from one observed spike vector."""
        self.rates += self.alpha * (actual.astype(np.float64) - self.rates)

    def reset(self) -> None:
        """Reset all channel rates to zero."""
        self.rates[:] = 0.0


def _predict_and_xor_context(
    spikes: np.ndarray[Any, Any], N: int, context_bits: int = 8
) -> tuple[np.ndarray[Any, Any], int]:
    """Context-model predict-XOR loop. Returns (errors, correct_count).

    Per-channel Markov predictor: hash last K spike states as context key,
    predict based on accumulated statistics for that context.
    Captures temporal patterns (bursts, oscillations, refractory periods)
    that EMA misses.
    """
    T = spikes.shape[0]
    errors = np.empty_like(spikes)
    correct = 0
    mask = (1 << context_bits) - 1

    # Per-channel: context register + prediction table
    contexts = np.zeros(N, dtype=np.int64)
    # Tables: context_key → [n_spikes_after, n_total]
    tables: list[dict[int, list[int]]] = [{} for _ in range(N)]

    for t in range(T):
        row = spikes[t]
        for ch in range(N):
            ctx = int(contexts[ch])
            table = tables[ch]

            # Predict from context statistics
            if ctx in table:
                n_spike, n_total = table[ctx]
                predicted = 1 if n_spike * 2 > n_total else 0
            else:
                predicted = 0

            actual = int(row[ch])
            err = actual ^ predicted
            errors[t, ch] = err
            if err == 0:
                correct += 1

            # Update context table
            if ctx not in table:
                table[ctx] = [0, 0]
            table[ctx][1] += 1
            if actual:
                table[ctx][0] += 1

            # Shift context register
            contexts[ch] = ((contexts[ch] << 1) | actual) & mask

    return errors, correct


def _xor_and_recover_context(
    errors: np.ndarray[Any, Any], N: int, context_bits: int = 8
) -> np.ndarray[Any, Any]:
    """Context-model XOR-recover loop for decoder."""
    T = errors.shape[0]
    spikes = np.empty((T, N), dtype=np.int8)
    mask = (1 << context_bits) - 1
    contexts = np.zeros(N, dtype=np.int64)
    tables: list[dict[int, list[int]]] = [{} for _ in range(N)]

    for t in range(T):
        for ch in range(N):
            ctx = int(contexts[ch])
            table = tables[ch]

            if ctx in table:
                n_spike, n_total = table[ctx]
                predicted = 1 if n_spike * 2 > n_total else 0
            else:
                predicted = 0

            actual = int(errors[t, ch]) ^ predicted
            spikes[t, ch] = actual

            if ctx not in table:
                table[ctx] = [0, 0]
            table[ctx][1] += 1
            if actual:
                table[ctx][0] += 1

            contexts[ch] = ((contexts[ch] << 1) | actual) & mask

    return spikes


def _predict_and_xor(
    spikes: np.ndarray[Any, Any], N: int, alpha: float, threshold: float
) -> tuple[np.ndarray[Any, Any], int]:
    """EMA predict-XOR loop. Returns (errors, correct_count)."""
    T = spikes.shape[0]
    rates = np.zeros(N, dtype=np.float64)
    errors = np.empty_like(spikes)
    correct = 0
    alpha_f = float(alpha)
    one_minus_alpha = 1.0 - alpha_f
    for t in range(T):
        row = spikes[t]
        predicted = (rates > threshold).view(np.int8)
        errors[t] = row ^ predicted
        correct += N - int(np.count_nonzero(errors[t]))
        rates *= one_minus_alpha
        rates += alpha_f * row
    return errors, correct


def _xor_and_recover(
    errors: np.ndarray[Any, Any], N: int, alpha: float, threshold: float
) -> np.ndarray[Any, Any]:
    """EMA XOR-recover loop for decoder."""
    T = errors.shape[0]
    rates = np.zeros(N, dtype=np.float64)
    spikes = np.empty((T, N), dtype=np.int8)
    alpha_f = float(alpha)
    one_minus_alpha = 1.0 - alpha_f
    for t in range(T):
        predicted = (rates > threshold).view(np.int8)
        row = errors[t] ^ predicted
        spikes[t] = row
        rates *= one_minus_alpha
        rates += alpha_f * row
    return spikes


# --- SC-native LFSR predictor (bit-true with sc_bitstream_encoder.v) ---

# LFSR-16 polynomial: x^16 + x^14 + x^13 + x^11 + 1
# Taps (0-indexed from MSB): 15, 13, 12, 10
_LFSR_MASK = 0xFFFF


def _lfsr16_step(reg: int) -> int:
    """One step of 16-bit Galois LFSR. Matches sc_bitstream_encoder.v."""
    feedback = ((reg >> 15) ^ (reg >> 13) ^ (reg >> 12) ^ (reg >> 10)) & 1
    return ((reg << 1) & _LFSR_MASK) | feedback


def _predict_and_xor_lfsr(
    spikes: np.ndarray[Any, Any],
    N: int,
    alpha_q8: int,
    seed: int,
) -> tuple[np.ndarray[Any, Any], int]:
    """LFSR-based predict-XOR loop. Bit-true with Verilog.

    Uses Q8.8 fixed-point rate tracking + LFSR comparator for prediction.
    Same polynomial and step semantics as sc_bitstream_encoder.v.

    Parameters
    ----------
    spikes : (T, N) int8
    N : int
    alpha_q8 : int
        Q8.8 smoothing factor. 1 = 1/256 ≈ 0.004.
    seed : int
        LFSR seed (non-zero, 16-bit).
    """
    T = spikes.shape[0]
    # Per-channel Q8.8 rate estimates (0-255 maps to 0.0-~1.0)
    rates_q8 = np.zeros(N, dtype=np.int32)
    errors = np.empty_like(spikes)
    correct = 0

    # Per-channel LFSR state (different seed per channel for decorrelation)
    lfsr_regs = np.array(
        [((seed + ch * 7919) & _LFSR_MASK) or 1 for ch in range(N)],
        dtype=np.int32,
    )

    for t in range(T):
        row = spikes[t]

        # Predict: LFSR < rate_q8 → predict spike (same as Verilog comparator)
        predicted = (lfsr_regs < rates_q8).astype(np.int8)

        # Step all LFSRs
        for ch in range(N):
            lfsr_regs[ch] = _lfsr16_step(int(lfsr_regs[ch]))

        # XOR
        errors[t] = row ^ predicted
        correct += N - int(np.count_nonzero(errors[t]))

        # Q8.8 EMA update: rate += alpha * (actual - rate) >> 8
        # Equivalent: rate = rate + alpha * (actual*256 - rate) >> 8
        for ch in range(N):
            target = 255 if row[ch] else 0
            rates_q8[ch] += (alpha_q8 * (target - rates_q8[ch])) >> 8
            rates_q8[ch] = max(0, min(255, rates_q8[ch]))

    return errors, correct


def _xor_and_recover_lfsr(
    errors: np.ndarray[Any, Any],
    N: int,
    alpha_q8: int,
    seed: int,
) -> np.ndarray[Any, Any]:
    """LFSR-based XOR-recover loop for decoder."""
    T = errors.shape[0]
    rates_q8 = np.zeros(N, dtype=np.int32)
    spikes = np.empty((T, N), dtype=np.int8)

    lfsr_regs = np.array(
        [((seed + ch * 7919) & _LFSR_MASK) or 1 for ch in range(N)],
        dtype=np.int32,
    )

    for t in range(T):
        predicted = (lfsr_regs < rates_q8).astype(np.int8)

        for ch in range(N):
            lfsr_regs[ch] = _lfsr16_step(int(lfsr_regs[ch]))

        row = errors[t] ^ predicted
        spikes[t] = row

        for ch in range(N):
            target = 255 if row[ch] else 0
            rates_q8[ch] += (alpha_q8 * (target - rates_q8[ch])) >> 8
            rates_q8[ch] = max(0, min(255, rates_q8[ch]))

    return spikes


class PredictiveSpikeCodec:
    """Predictive spike codec: compress prediction errors, not raw spikes.

    Four predictor modes:
        'ema' (default): float EMA rate tracking + threshold comparison.
        'lfsr': Q8.8 fixed-point rate + LFSR comparator. Bit-true with
                sc_bitstream_encoder.v — maps directly to Verilog RTL.
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
        2. ISI-compress the error matrix (sparser than raw spikes)
        3. Pack with header (predictor params for decoder sync)

    Parameters
    ----------
    alpha : float
        EMA smoothing factor (ema mode). Ignored in lfsr/context mode.
    threshold : float
        Spike prediction threshold (ema mode). Ignored in lfsr/context mode.
    predictor : str
        'ema', 'lfsr', or 'context'.
    alpha_q8 : int
        Q8.8 smoothing factor for lfsr mode. 1 = 1/256 ≈ 0.004.
    seed : int
        LFSR seed for lfsr mode (non-zero, 16-bit).
    context_bits : int
        Context history length for context mode (default 8 = last 8 spikes).
    base_mode : str
        'lossless' or 'lossy' for the underlying ISI codec.
    timing_precision : int
        For lossy mode: quantize timing resolution.
    """

    HEADER_MAGIC = b"PSCX"  # Predictive Spike Codec XOR (EMA)
    HEADER_MAGIC_LFSR = b"PSCL"  # Predictive Spike Codec LFSR
    HEADER_MAGIC_CTX = b"PSCC"  # Predictive Spike Codec Context
    HEADER_MAGIC_WM = b"PSCW"  # Predictive Spike Codec World Model

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
    ):
        self.alpha = alpha
        self.threshold = threshold
        self.predictor = predictor
        self.alpha_q8 = alpha_q8
        self.seed = seed
        self.context_bits = context_bits
        # Context predictor benefits from Huffman backend (sparse scattered errors)
        entropy = "huffman" if predictor in ("context", "world_model") else "auto"
        self.base_codec = SpikeCodec(
            mode=base_mode,
            timing_precision=timing_precision,
            entropy=entropy,
        )

    def compress(self, spikes: np.ndarray[Any, Any]) -> tuple[bytes, PredictiveCompressionResult]:
        """Compress spike raster using predictive error coding.

        Parameters
        ----------
        spikes : ndarray of shape (T, N), binary (int8 or bool)

        Returns
        -------
        (compressed_bytes, PredictiveCompressionResult)
        """
        import struct

        spikes = np.asarray(spikes, dtype=np.int8)
        T, N = spikes.shape
        original_bits = T * N

        if self.predictor == "world_model":
            errors, correct_predictions = predict_and_xor_world_model(
                spikes,
                N,
                history_len=self.context_bits,
                lr=self.alpha,
                threshold=self.threshold,
                seed=self.seed,
            )
            error_data, _ = self.base_codec.compress(errors)
            header = self.HEADER_MAGIC_WM + struct.pack(
                "!BdH",
                self.context_bits,
                self.alpha,
                self.seed,
            )
        elif self.predictor == "context":
            errors, correct_predictions = _predict_and_xor_context(
                spikes,
                N,
                self.context_bits,
            )
            error_data, _ = self.base_codec.compress(errors)
            header = self.HEADER_MAGIC_CTX + struct.pack("!B", self.context_bits)
        elif self.predictor == "lfsr":
            if _HAS_RUST:  # pragma: no cover
                flat = np.ascontiguousarray(spikes).ravel()
                err_flat, correct_predictions = _rust_predict_lfsr(
                    flat,
                    N,
                    self.alpha_q8,
                    self.seed,
                )
                errors = np.asarray(err_flat).reshape(T, N)
            else:
                errors, correct_predictions = _predict_and_xor_lfsr(
                    spikes,
                    N,
                    self.alpha_q8,
                    self.seed,
                )
            error_data, _ = self.base_codec.compress(errors)
            header = self.HEADER_MAGIC_LFSR + struct.pack("!HH", self.alpha_q8, self.seed)
        else:
            if _HAS_RUST:  # pragma: no cover
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
            error_data, _ = self.base_codec.compress(errors)
            header = self.HEADER_MAGIC + struct.pack("!dd", self.alpha, self.threshold)

        encoded = header + error_data
        compressed_bits = len(encoded) * 8
        ratio = original_bits / max(compressed_bits, 1)

        return encoded, PredictiveCompressionResult(
            original_bits=original_bits,
            compressed_bits=compressed_bits,
            compression_ratio=ratio,
            n_spikes=int(np.sum(spikes)),
            n_neurons=N,
            n_timesteps=T,
            lossless=self.base_codec.mode == "lossless",
            prediction_accuracy=correct_predictions / max(T * N, 1),
            error_sparsity=1.0 - (int(np.sum(errors)) / max(T * N, 1)),
            predictor_type=self.predictor,
        )

    def decompress(self, data: bytes, T: int, N: int) -> np.ndarray[Any, Any]:
        """Decompress to spike raster.

        Runs identical predictor on decoder side. XOR(error, predicted)
        recovers original spikes. Predictor type auto-detected from header.

        Parameters
        ----------
        data : bytes
            Compressed data from compress().
        T, N : int
            Original dimensions.

        Returns
        -------
        ndarray of shape (T, N), int8
        """
        import struct

        magic = data[:4]

        if magic == self.HEADER_MAGIC_WM:
            history_len = data[4]
            alpha, seed = struct.unpack("!dH", data[5:15])
            error_data = data[15:]
            errors = self.base_codec.decompress(error_data, T, N)
            return xor_and_recover_world_model(
                errors,
                N,
                history_len=history_len,
                lr=alpha,
                seed=seed,
            )

        if magic == self.HEADER_MAGIC_CTX:
            context_bits = data[4]
            error_data = data[5:]
            errors = self.base_codec.decompress(error_data, T, N)
            return _xor_and_recover_context(errors, N, context_bits)

        if magic == self.HEADER_MAGIC_LFSR:
            alpha_q8, seed = struct.unpack("!HH", data[4:8])
            error_data = data[8:]
            errors = self.base_codec.decompress(error_data, T, N)
            if _HAS_RUST:  # pragma: no cover
                flat = np.ascontiguousarray(errors).ravel()
                rec = np.asarray(_rust_recover_lfsr(flat, N, alpha_q8, seed))
                return rec.reshape(T, N)
            return _xor_and_recover_lfsr(errors, N, alpha_q8, seed)

        if magic == self.HEADER_MAGIC:
            alpha, threshold = struct.unpack("!dd", data[4:20])
            error_data = data[20:]
            errors = self.base_codec.decompress(error_data, T, N)
            if _HAS_RUST:  # pragma: no cover
                flat = np.ascontiguousarray(errors).ravel()
                rec = np.asarray(_rust_recover_ema(flat, N, alpha, threshold))
                return rec.reshape(T, N)
            return _xor_and_recover(errors, N, alpha, threshold)

        raise ValueError(
            f"Invalid header magic: {magic!r}, expected {self.HEADER_MAGIC!r} or {self.HEADER_MAGIC_LFSR!r}"
        )
