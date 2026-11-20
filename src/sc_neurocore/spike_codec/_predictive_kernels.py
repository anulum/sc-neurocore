# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reference kernels for predictive spike packets

"""Compute the EMA, fixed-point LFSR and context XOR transforms."""

from __future__ import annotations

from typing import Any

import numpy as np


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


# Fixed-point LFSR predictor

# LFSR-16 polynomial: x^16 + x^14 + x^13 + x^11 + 1
# Taps (0-indexed from MSB): 15, 13, 12, 10
_LFSR_MASK = 0xFFFF


def _lfsr16_step(reg: int) -> int:
    """Advance the 16-bit register using the encoder primitive’s tap recurrence."""
    feedback = ((reg >> 15) ^ (reg >> 13) ^ (reg >> 12) ^ (reg >> 10)) & 1
    return ((reg << 1) & _LFSR_MASK) | feedback


def _predict_and_xor_lfsr(
    spikes: np.ndarray[Any, Any],
    N: int,
    alpha_q8: int,
    seed: int,
) -> tuple[np.ndarray[Any, Any], int]:
    """Predict and XOR using fixed-point rates and per-channel LFSR state.

    The register recurrence matches the 16-bit generator in
    sc_bitstream_encoder.v. Rate tracking is part of the codec.

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

        # Predict from the unsigned register and the running fixed-point rate.
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
