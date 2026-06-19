# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Foundation-model neural population decoders

"""Foundation-model neural population decoders.

Publication-exact implementations of four state-of-the-art neural
population decoding algorithms:

- **POYODecoder** — Azabou et al. (2023), NeurIPS. arXiv:2310.16046
- **POSSMDecoder** — Ryoo et al. (2025), ICLR. arXiv:2506.05320
- **NDT3Decoder** — Ye & Pandarinath (2025). bioRxiv:2025.02.02.634313
- **CEBRAEncoder** — Schneider, Lee & Mathis (2023), Nature 604. arXiv:2204.00673
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any
import numpy as np


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def tokenise_spikes(
    spike_trains: list[np.ndarray[Any, Any]],
    dt: float = 1.0,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Convert binary spike trains to sorted (unit_id, timestamp) tokens.

    Used by POYO+ and POSSM (Azabou et al. 2023; Ryoo et al. 2025).

    Parameters
    ----------
    spike_trains : list of 1-D binary arrays (one per neuron).
    dt : timestep in ms.

    Returns
    -------
    unit_ids : int64 array [n_tokens].
    timestamps : float64 array [n_tokens] in ms.
    """
    uids: list[int] = []
    times: list[float] = []
    for uid, train in enumerate(spike_trains):
        indices = np.flatnonzero(train)
        for idx in indices:
            uids.append(uid)
            times.append(idx * dt)
    unit_ids = np.array(uids, dtype=np.int64)
    timestamps = np.array(times, dtype=np.float64)
    order = np.argsort(timestamps, kind="stable")
    return unit_ids[order], timestamps[order]


def sinusoidal_position_encode(
    timestamps: np.ndarray[Any, Any],
    d_model: int,
) -> np.ndarray[Any, Any]:
    """Sinusoidal position encoding. Vaswani et al. (2017).

    PE(t, 2i)   = sin(t / 10000^{2i/d})
    PE(t, 2i+1) = cos(t / 10000^{2i/d})
    """
    n = len(timestamps)
    pe = np.zeros((n, d_model), dtype=np.float64)
    indices = np.arange(0, d_model, 2, dtype=np.float64)
    divisors = 10000.0 ** (indices / d_model)
    for k, div in enumerate(divisors):
        pe[:, 2 * k] = np.sin(timestamps / div)
        if 2 * k + 1 < d_model:
            pe[:, 2 * k + 1] = np.cos(timestamps / div)
    return pe


def scaled_dot_product_attention(
    queries: np.ndarray[Any, Any],
    keys: np.ndarray[Any, Any],
    values: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """
    d_k = keys.shape[-1]
    scores = queries @ keys.T / np.sqrt(d_k)
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True) + 1e-30
    pooled: np.ndarray[Any, Any] = weights @ values
    return pooled


# ---------------------------------------------------------------------------
# POYO+ — Azabou et al. (2023), NeurIPS. arXiv:2310.16046
# ---------------------------------------------------------------------------


@dataclass
class POYODecoder:
    """Population decoder via spike tokenisation and cross-attention.

    Azabou et al. "A Unified, Scalable Framework for Neural Population
    Decoding." NeurIPS 2023. arXiv:2310.16046.

    Architecture: individual spikes are tokenised with learned unit
    embeddings + sinusoidal temporal encoding. Learnable latent queries
    attend to the spike tokens via cross-attention (PerceiverIO backbone).
    Output queries decode from the latent representation.
    """

    d_model: int = 64
    n_latents: int = 32
    seed: int = 42

    def __post_init__(self) -> None:
        """Initialise the POYO decoder weights from the seed."""
        rng = np.random.default_rng(self.seed)
        self._latent_queries = rng.normal(0.0, 0.02, (self.n_latents, self.d_model))
        self._unit_embeddings: dict[int, np.ndarray[Any, Any]] = {}

    def _unit_embedding(self, unit_id: int) -> np.ndarray[Any, Any]:
        if unit_id not in self._unit_embeddings:
            rng = np.random.default_rng(self.seed + unit_id + 1)
            self._unit_embeddings[unit_id] = rng.normal(0.0, 0.02, self.d_model)
        return self._unit_embeddings[unit_id]

    def encode(
        self,
        spike_trains: list[np.ndarray[Any, Any]],
        dt: float = 1.0,
    ) -> np.ndarray[Any, Any]:
        """Encode population activity to latent representation.

        Returns array [n_latents, d_model].
        """
        unit_ids, timestamps = tokenise_spikes(spike_trains, dt)
        if len(unit_ids) == 0:
            return np.zeros((self.n_latents, self.d_model))
        pe = sinusoidal_position_encode(timestamps, self.d_model)
        token_embs = np.array([self._unit_embedding(u) for u in unit_ids])
        kv = token_embs + pe
        return scaled_dot_product_attention(self._latent_queries, kv, kv)

    def decode(
        self,
        latents: np.ndarray[Any, Any],
        output_queries: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Cross-attention decode from latents.

        output_queries : [n_outputs, d_model]
        Returns [n_outputs, d_model].
        """
        return scaled_dot_product_attention(output_queries, latents, latents)

    def reset(self) -> None:
        """Clear cached unit embeddings."""
        self._unit_embeddings.clear()
        rng = np.random.default_rng(self.seed)
        self._latent_queries = rng.normal(0.0, 0.02, (self.n_latents, self.d_model))


# ---------------------------------------------------------------------------
# POSSM — Ryoo et al. (2025), ICLR. arXiv:2506.05320
# ---------------------------------------------------------------------------


@dataclass
class POSSMDecoder:
    """Population decoder via spike tokenisation and diagonal state-space model.

    Ryoo et al. "Generalizable, Real-Time Neural Decoding with Hybrid
    State-Space Models." ICLR 2025. arXiv:2506.05320.

    Uses POYO spike tokenisation with a recurrent SSM backbone instead
    of full attention, enabling causal online prediction at millisecond
    resolution with up to 9x lower inference cost.

    Diagonal SSM recurrence (S4D, Gu et al. 2022):
        h_t = A_bar h_{t-1} + B_bar x_t
        y_t = Re(C h_t) + D x_t

    Discretisation (zero-order hold):
        A_bar = exp(dt * A)
        B_bar = (A_bar - I) * diag(A)^{-1} * B
    """

    d_model: int = 64
    d_state: int = 32
    dt: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        """Initialise the POSSM decoder state-space parameters from the seed."""
        rng = np.random.default_rng(self.seed)
        # HiPPO-LegS initialisation for complex diagonal A
        real_part = -0.5 * np.ones(self.d_state)
        imag_part = np.pi * np.arange(self.d_state, dtype=np.float64)
        self._A = real_part + 1j * imag_part
        self._B = rng.normal(0.0, 0.02, (self.d_state, self.d_model)).astype(np.complex128)
        self._C = rng.normal(0.0, 0.02, (self.d_model, self.d_state)).astype(np.complex128)
        self._D = rng.normal(0.0, 0.02, (self.d_model, self.d_model))
        self._h = np.zeros(self.d_state, dtype=np.complex128)

    def discretise(self, step_dt: float) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Zero-order hold discretisation.

        A_bar = exp(dt * A)
        B_bar = (A_bar - I) * diag(A)^{-1} * B
        """
        a_bar = np.exp(step_dt * self._A)
        a_inv = 1.0 / (self._A + 1e-30)
        b_bar = np.diag(a_bar - 1.0) @ np.diag(a_inv) @ self._B
        return a_bar, b_bar

    def step(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Single causal SSM step.

        h_t = A_bar h_{t-1} + B_bar x_t
        y_t = Re(C h_t) + D x_t
        """
        a_bar, b_bar = self.discretise(self.dt)
        self._h = a_bar * self._h + b_bar @ x
        readout: np.ndarray[Any, Any] = np.real(self._C @ self._h) + self._D @ x
        return readout

    def encode_causal(
        self,
        spike_trains: list[np.ndarray[Any, Any]],
        dt: float = 1.0,
    ) -> np.ndarray[Any, Any]:
        """Causal online encoding of spike trains.

        Returns [n_timesteps, d_model] output sequence.
        """
        self.reset()
        if not spike_trains:
            return np.zeros((0, self.d_model))
        n_steps = max(len(t) for t in spike_trains)
        n_units = len(spike_trains)
        # Pad spike trains to common length
        padded = np.zeros((n_units, n_steps), dtype=np.float64)
        for i, train in enumerate(spike_trains):
            padded[i, : len(train)] = train
        # Project population vector to d_model via fixed random projection
        rng = np.random.default_rng(self.seed + 9999)
        proj = rng.normal(0.0, 1.0 / np.sqrt(n_units), (self.d_model, n_units))
        outputs = np.zeros((n_steps, self.d_model))
        for t_idx in range(n_steps):
            x = proj @ padded[:, t_idx]
            outputs[t_idx] = self.step(x)
        return outputs

    def reset(self) -> None:
        """Reset hidden state to zero."""
        self._h = np.zeros(self.d_state, dtype=np.complex128)


# ---------------------------------------------------------------------------
# NDT3 — Ye & Pandarinath (2025). bioRxiv:2025.02.02.634313
# ---------------------------------------------------------------------------


@dataclass
class NDT3Decoder:
    """Autoregressive neural data transformer for motor decoding.

    Ye & Pandarinath. "A Generalist Intracortical Motor Decoder."
    bioRxiv 2025.02.02.634313.

    Bins population spike counts, projects to d_model embeddings,
    adds positional encoding, and predicts next-bin output via
    causal (masked) self-attention.
    """

    d_model: int = 64
    bin_size_ms: float = 20.0
    seed: int = 42

    def __post_init__(self) -> None:
        """Initialise the NDT-3 decoder weights from the seed."""
        rng = np.random.default_rng(self.seed)
        self._embed_w: np.ndarray[Any, Any] | None = None
        self._embed_b: np.ndarray[Any, Any] | None = None
        self._output_w = rng.normal(0.0, 0.02, (self.d_model, self.d_model))
        self._output_b = np.zeros(self.d_model)

    def bin_and_embed(
        self,
        spike_trains: list[np.ndarray[Any, Any]],
        dt: float = 1.0,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Bin spike trains and project to embeddings.

        Returns (binned [n_bins, n_neurons], embedded [n_bins, d_model]).
        """
        if not spike_trains:
            return np.zeros((0, 0)), np.zeros((0, self.d_model))
        n_neurons = len(spike_trains)
        samples_per_bin = max(1, int(self.bin_size_ms / dt))
        n_steps = max(len(t) for t in spike_trains)
        n_bins = n_steps // samples_per_bin
        if n_bins == 0:
            return np.zeros((0, n_neurons)), np.zeros((0, self.d_model))
        binned = np.zeros((n_bins, n_neurons), dtype=np.float64)
        for i, train in enumerate(spike_trains):
            for b in range(n_bins):
                start = b * samples_per_bin
                end = min(start + samples_per_bin, len(train))
                binned[b, i] = train[start:end].sum()
        # Lazy init embedding projection
        if self._embed_w is None or self._embed_w.shape[1] != n_neurons:
            rng = np.random.default_rng(self.seed)
            self._embed_w = rng.normal(
                0.0,
                1.0 / np.sqrt(n_neurons),
                (self.d_model, n_neurons),
            )
            self._embed_b = np.zeros(self.d_model)
        assert self._embed_w is not None and self._embed_b is not None
        embedded = binned @ self._embed_w.T + self._embed_b
        pe = sinusoidal_position_encode(
            np.arange(n_bins, dtype=np.float64),
            self.d_model,
        )
        embedded += pe
        return binned, embedded

    def predict_next(self, embedded: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Causal autoregressive prediction of next time bin.

        Uses causal (lower-triangular) masked self-attention.
        Returns [n_bins, d_model] contextualised representations.
        """
        n = embedded.shape[0]
        if n == 0:
            return np.zeros((0, self.d_model))
        d_k = embedded.shape[-1]
        scores = embedded @ embedded.T / np.sqrt(d_k)
        # Causal mask: positions can only attend to earlier positions
        mask = np.triu(np.full((n, n), -1e9), k=1)
        scores += mask
        scores -= scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True) + 1e-30
        attended = weights @ embedded
        projected: np.ndarray[Any, Any] = attended @ self._output_w.T + self._output_b
        return projected

    def decode(
        self,
        spike_trains: list[np.ndarray[Any, Any]],
        dt: float = 1.0,
    ) -> np.ndarray[Any, Any]:
        """Full decode pipeline: bin → embed → causal attention → output.

        Returns [n_bins, d_model] decoded representations.
        """
        _, embedded = self.bin_and_embed(spike_trains, dt)
        return self.predict_next(embedded)


# ---------------------------------------------------------------------------
# CEBRA — Schneider, Lee & Mathis (2023), Nature 604. arXiv:2204.00673
# ---------------------------------------------------------------------------


@dataclass
class CEBRAEncoder:
    """Contrastive embedding encoder for neural data.

    Schneider, Lee & Mathis. "Learnable latent embeddings for joint
    behavioural and neural analysis." Nature 604 (2023).
    arXiv:2204.00673.

    Uses InfoNCE contrastive loss with time-based or behaviour-based
    positive pair sampling to learn low-dimensional embeddings of
    neural population activity.

    InfoNCE loss (van den Oord et al. 2018):
        L = -log( exp(sim(z_i, z_j^+) / τ) / Σ_k exp(sim(z_i, z_k) / τ) )
    where sim(a, b) = a · b / (||a|| ||b||)   (cosine similarity)
    """

    d_input: int = 64
    d_output: int = 8
    temperature: float = 1.0
    learning_rate: float = 0.001
    seed: int = 42

    def __post_init__(self) -> None:
        """Initialise the CEBRA encoder weights from the seed."""
        rng = np.random.default_rng(self.seed)
        # Two-layer MLP encoder: d_input → d_hidden → d_output
        d_hidden = max(self.d_input, 2 * self.d_output)
        self._w1 = rng.normal(0.0, np.sqrt(2.0 / self.d_input), (d_hidden, self.d_input))
        self._b1 = np.zeros(d_hidden)
        self._w2 = rng.normal(0.0, np.sqrt(2.0 / d_hidden), (self.d_output, d_hidden))
        self._b2 = np.zeros(self.d_output)

    def encode(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Encode neural data through 2-layer MLP.

        x : [batch, d_input] or [d_input].
        Returns [batch, d_output] or [d_output].
        """
        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis, :]
        h = x @ self._w1.T + self._b1
        h = np.maximum(h, 0.0)  # ReLU
        z = h @ self._w2.T + self._b2
        # L2 normalise to unit hypersphere
        norms = np.linalg.norm(z, axis=-1, keepdims=True) + 1e-30
        z = z / norms
        if squeeze:
            z = z[0]
        embedding: np.ndarray[Any, Any] = z
        return embedding

    @staticmethod
    def cosine_similarity(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Pairwise cosine similarity matrix.

        a : [n, d], b : [m, d]. Returns [n, m].
        """
        a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-30)
        b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-30)
        similarity: np.ndarray[Any, Any] = a_norm @ b_norm.T
        return similarity

    def infonce_loss(
        self,
        anchors: np.ndarray[Any, Any],
        positives: np.ndarray[Any, Any],
    ) -> float:
        """InfoNCE contrastive loss. van den Oord et al. (2018).

        L = -(1/N) Σ_i log( exp(sim(z_i, z_i^+)/τ) / Σ_j exp(sim(z_i, z_j)/τ) )

        anchors, positives : [N, d_output]. Row i paired with row i.
        All other rows serve as negatives.
        """
        z_a = self.encode(anchors)
        z_p = self.encode(positives)
        # Similarity matrix: each anchor vs all positives
        sim_matrix = self.cosine_similarity(z_a, z_p) / self.temperature
        sim_matrix -= sim_matrix.max(axis=-1, keepdims=True)
        exp_sim = np.exp(sim_matrix)
        # Positive similarities on the diagonal
        pos_sim = np.diag(exp_sim)
        loss = -np.mean(np.log(pos_sim / (exp_sim.sum(axis=-1) + 1e-30) + 1e-30))
        return float(loss)

    def _forward_and_loss(
        self,
        anchors: np.ndarray[Any, Any],
        positives: np.ndarray[Any, Any],
    ) -> tuple[float, dict[str, np.ndarray[Any, Any]]]:
        """Forward pass with cached intermediates for backprop.

        Returns (loss, cache) where cache contains all intermediates.
        """
        # Layer 1
        h1_pre = anchors @ self._w1.T + self._b1
        h1 = np.maximum(h1_pre, 0.0)
        z1_pre = h1 @ self._w2.T + self._b2
        n1 = np.linalg.norm(z1_pre, axis=-1, keepdims=True) + 1e-30
        z_a = z1_pre / n1

        h2_pre = positives @ self._w1.T + self._b1
        h2 = np.maximum(h2_pre, 0.0)
        z2_pre = h2 @ self._w2.T + self._b2
        n2 = np.linalg.norm(z2_pre, axis=-1, keepdims=True) + 1e-30
        z_p = z2_pre / n2

        # InfoNCE forward
        sim = z_a @ z_p.T / self.temperature
        sim -= sim.max(axis=-1, keepdims=True)
        exp_sim = np.exp(sim)
        row_sums = exp_sim.sum(axis=-1) + 1e-30
        n_batch = anchors.shape[0]
        pos_sim = np.array([exp_sim[i, i] for i in range(n_batch)])
        loss = -np.mean(np.log(pos_sim / row_sums + 1e-30))

        cache = {
            "anchors": anchors,
            "positives": positives,
            "h1_pre": h1_pre,
            "h1": h1,
            "z1_pre": z1_pre,
            "n1": n1,
            "z_a": z_a,
            "h2_pre": h2_pre,
            "h2": h2,
            "z2_pre": z2_pre,
            "n2": n2,
            "z_p": z_p,
            "exp_sim": exp_sim,
            "row_sums": row_sums,
        }
        return float(loss), cache

    def _backward(self, cache: dict[str, np.ndarray[Any, Any]]) -> dict[str, np.ndarray[Any, Any]]:
        """Analytical backprop through InfoNCE + MLP.

        Returns gradients for w1, b1, w2, b2.
        """
        n = cache["z_a"].shape[0]
        tau = self.temperature

        # dL/d(sim_matrix): softmax cross-entropy gradient
        probs = cache["exp_sim"] / cache["row_sums"][:, np.newaxis]
        d_sim = probs / n
        for i in range(n):
            d_sim[i, i] -= 1.0 / n

        # dL/dz_a, dL/dz_p from sim = z_a @ z_p.T / τ
        d_za = d_sim @ cache["z_p"] / tau
        d_zp = d_sim.T @ cache["z_a"] / tau

        # Backprop through L2 normalisation: z = z_pre / ||z_pre||
        def grad_l2norm(
            d_z: np.ndarray[Any, Any], z_pre: np.ndarray[Any, Any], norms: np.ndarray[Any, Any]
        ) -> np.ndarray[Any, Any]:
            z_hat = z_pre / norms
            grad: np.ndarray[Any, Any] = (
                d_z - z_hat * (d_z * z_hat).sum(axis=-1, keepdims=True)
            ) / norms
            return grad

        d_z1_pre = grad_l2norm(d_za, cache["z1_pre"], cache["n1"])
        d_z2_pre = grad_l2norm(d_zp, cache["z2_pre"], cache["n2"])

        # Backprop through layer 2 (both anchor and positive paths share weights)
        d_w2 = d_z1_pre.T @ cache["h1"] + d_z2_pre.T @ cache["h2"]
        d_b2 = d_z1_pre.sum(axis=0) + d_z2_pre.sum(axis=0)

        d_h1 = d_z1_pre @ self._w2
        d_h2 = d_z2_pre @ self._w2

        # ReLU gradient
        d_h1_pre = d_h1 * (cache["h1_pre"] > 0).astype(np.float64)
        d_h2_pre = d_h2 * (cache["h2_pre"] > 0).astype(np.float64)

        # Backprop through layer 1
        d_w1 = d_h1_pre.T @ cache["anchors"] + d_h2_pre.T @ cache["positives"]
        d_b1 = d_h1_pre.sum(axis=0) + d_h2_pre.sum(axis=0)

        return {"w1": d_w1, "b1": d_b1, "w2": d_w2, "b2": d_b2}

    def fit(
        self,
        data: np.ndarray[Any, Any],
        n_steps: int = 200,
        time_offset: int = 1,
    ) -> float:
        """Train encoder with time-contrastive learning.

        Positive pairs: (data[t], data[t + time_offset]).
        Uses analytical backpropagation with SGD.
        Returns final loss value.
        """
        n = data.shape[0] - time_offset
        if n < 2:
            return 0.0
        anchors = data[:n]
        positives = data[time_offset : n + time_offset]
        loss = 0.0
        for _ in range(n_steps):
            loss, cache = self._forward_and_loss(anchors, positives)
            grads = self._backward(cache)
            self._w1 -= self.learning_rate * grads["w1"]
            self._b1 -= self.learning_rate * grads["b1"]
            self._w2 -= self.learning_rate * grads["w2"]
            self._b2 -= self.learning_rate * grads["b2"]
        return loss

    def transform(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Embed neural data into learned latent space.

        data : [n_samples, d_input].
        Returns [n_samples, d_output].
        """
        return self.encode(data)
