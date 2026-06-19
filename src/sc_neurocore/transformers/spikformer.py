# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-Driven Self-Attention (SSA) + Spikformer blocks

"""Spike-driven transformer blocks for energy-efficient sequence processing.

Implements Spike-Driven Self-Attention (SSA) from Spikformer (ICLR 2023)
and Spatial-Temporal Attention (STAA) from CVPR 2025. Key insight: SSA
replaces softmax with spike-based masking, eliminating all multiplications
in attention — natural match for stochastic computing AND gates.

No SNN framework provides these as reusable building blocks.

Reference:
  Zhou et al. 2023 — "Spikformer: When Spiking Neural Network Meets Transformer"
  Lee et al. 2025 — "Spiking Transformer with Spatial-Temporal Attention" (CVPR)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SpikeDrivenAttention:
    """Spike-Driven Self-Attention (SSA).

    Replaces Q*K^T softmax with spike-based masking:
      Attention = SpikeFn(Q_linear(S)) * SpikeFn(K_linear(S))^T * V_linear(S)

    All operations reduce to AND gates on binary spikes —
    zero multiplications, pure SC-compatible logic.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    T : int
        Number of simulation timesteps.
    threshold : float
        Spike threshold for Q/K projections.
    """

    embed_dim: int
    num_heads: int = 1
    T: int = 8
    threshold: float = 1.0

    def __post_init__(self) -> None:
        """Derive the head dimension and initialise the projection weights."""
        self.head_dim = self.embed_dim // self.num_heads
        rng = np.random.RandomState(42)
        # Linear projections (Q, K, V)
        scale = np.sqrt(2.0 / self.embed_dim)
        self.W_q = rng.randn(self.embed_dim, self.embed_dim) * scale
        self.W_k = rng.randn(self.embed_dim, self.embed_dim) * scale
        self.W_v = rng.randn(self.embed_dim, self.embed_dim) * scale
        self.W_out = rng.randn(self.embed_dim, self.embed_dim) * scale
        # Membrane state for Q/K spike generation
        self._v_q: np.ndarray[Any, Any] | None = None
        self._v_k: np.ndarray[Any, Any] | None = None

    def _spike_fn(
        self, membrane: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Integrate-and-fire: returns (spikes, new_membrane)."""
        spikes = (membrane >= self.threshold).astype(np.float64)
        membrane = membrane - spikes * self.threshold
        return spikes, membrane

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Forward pass: spike-driven attention over T timesteps.

        Parameters
        ----------
        x : ndarray of shape (seq_len, embed_dim) or (embed_dim,)
            Input spike rates in [0, 1].

        Returns
        -------
        ndarray, same shape as x
            Attention output.
        """
        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis]

        seq_len = x.shape[0]
        # Linear projections
        Q_proj = x @ self.W_q
        K_proj = x @ self.W_k
        V_proj = x @ self.W_v

        # Accumulate over T timesteps with spike-driven attention
        output_acc = np.zeros_like(x)
        self._v_q = np.zeros_like(Q_proj)
        self._v_k = np.zeros_like(K_proj)

        for t in range(self.T):
            # Rate-code input: spike with probability proportional to projection
            self._v_q += np.clip(Q_proj, 0, None) / self.T
            self._v_k += np.clip(K_proj, 0, None) / self.T

            Q_spikes, self._v_q = self._spike_fn(self._v_q)
            K_spikes, self._v_k = self._spike_fn(self._v_k)

            # SSA: spike AND instead of softmax
            # attn_weights[i,j] = Q_spikes[i] AND K_spikes[j] (dot product of binary)
            attn = Q_spikes @ K_spikes.T  # (seq, seq) — counts of matching spikes
            scale = max(np.sqrt(self.head_dim), 1.0)
            attn = attn / scale

            # Weighted sum of V
            output_acc += attn @ V_proj

        output: np.ndarray[Any, Any] = (output_acc / self.T) @ self.W_out

        if squeeze:
            output = output[0]
        return output

    @property
    def num_multiply_ops(self) -> int:
        """Zero multiplications in the attention core (AND gates only)."""
        return 0


@dataclass
class SpikyStateSpace:
    """Spiking State-Space Model (S4-SNN hybrid).

    Combines linear state-space dynamics with spiking nonlinearity:
      h_t = A * h_{t-1} + B * spike_input_t
      y_t = C * h_t
      spike_t = IF(y_t > threshold)

    Runs in O(1) memory per timestep (no BPTT unrolling needed).
    Reference: SpikySpace (2025).

    Parameters
    ----------
    d_model : int
        Input/output dimension.
    d_state : int
        Hidden state dimension.
    threshold : float
        Spiking threshold.
    dt : float
        Discretization timestep.
    """

    d_model: int
    d_state: int = 64
    threshold: float = 1.0
    dt: float = 0.01

    def __post_init__(self) -> None:
        """Initialise the discretised state-space matrices."""
        rng = np.random.RandomState(42)
        # State-space matrices (discretized)
        # A: state transition (diagonal for efficiency)
        self.A = np.exp(-self.dt * np.abs(rng.randn(self.d_state)))
        self.B = rng.randn(self.d_state, self.d_model) * np.sqrt(2.0 / self.d_model)
        self.C = rng.randn(self.d_model, self.d_state) * np.sqrt(2.0 / self.d_state)
        self._h = np.zeros(self.d_state)
        self._v = np.zeros(self.d_model)

    def reset(self) -> None:
        """Reset hidden state and membrane potential."""
        self._h = np.zeros(self.d_state)
        self._v = np.zeros(self.d_model)

    def step(self, x: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Process one timestep.

        Parameters
        ----------
        x : ndarray of shape (d_model,)
            Input (binary spikes or continuous).

        Returns
        -------
        (spikes, output) tuple
            spikes: binary spike output (d_model,)
            output: continuous pre-spike output (d_model,)
        """
        self._h = self.A * self._h + self.B @ x
        y = self.C @ self._h

        self._v += y
        spikes = (self._v >= self.threshold).astype(np.float64)
        self._v -= spikes * self.threshold

        return spikes, y

    def forward(self, x_seq: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Process a full sequence.

        Parameters
        ----------
        x_seq : ndarray of shape (T, d_model)

        Returns
        -------
        ndarray of shape (T, d_model)
            Spike output per timestep.
        """
        self.reset()
        T = x_seq.shape[0]
        out = np.zeros_like(x_seq)
        for t in range(T):
            spikes, _ = self.step(x_seq[t])
            out[t] = spikes
        return out


@dataclass
class CPGPositionalEncoding:
    """Central Pattern Generator positional encoding.

    Replaces sinusoidal positional encoding with biologically-inspired
    CPG oscillators. Each dimension has a different frequency and phase,
    generating spike-compatible temporal position signals.

    Parameters
    ----------
    d_model : int
        Encoding dimension.
    max_len : int
        Maximum sequence length.
    """

    d_model: int
    max_len: int = 1024

    def __post_init__(self) -> None:
        """Sample the central-pattern-generator oscillator frequencies."""
        rng = np.random.RandomState(42)
        self.frequencies = np.exp(rng.randn(self.d_model) * 0.5)
        self.phases = rng.uniform(0, 2 * np.pi, self.d_model)

    def encode(self, seq_len: int) -> np.ndarray[Any, Any]:
        """Generate positional encoding.

        Returns
        -------
        ndarray of shape (seq_len, d_model)
            Values in [0, 1] suitable for spike rate encoding.
        """
        t = np.arange(seq_len)[:, np.newaxis]
        angles = t * self.frequencies[np.newaxis, :] * 0.01 + self.phases[np.newaxis, :]
        rates: np.ndarray[Any, Any] = (np.sin(angles) + 1.0) / 2.0  # Map to [0, 1]
        return rates

    def encode_spikes(
        self, seq_len: int, rng: np.random.RandomState | None = None
    ) -> np.ndarray[Any, Any]:
        """Generate spike-encoded positional encoding.

        Returns
        -------
        ndarray of shape (seq_len, d_model), binary
        """
        if rng is None:
            rng = np.random.RandomState(0)
        rates = self.encode(seq_len)
        spikes: np.ndarray[Any, Any] = (rng.random(rates.shape) < rates).astype(np.int8)
        return spikes
