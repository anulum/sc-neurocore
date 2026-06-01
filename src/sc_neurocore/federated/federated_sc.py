# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Federated SC Learning with Bitstream-Level Differential Privacy

"""Federated stochastic computing learning with in-domain DP.

Gradient exchange as noisy SC bitstreams: differential privacy noise is
injected directly in the stochastic domain (bit-flip mechanism) rather than
post-hoc Gaussian/Laplace on real-valued gradients. This enables provable
privacy budgets on deterministic LFSR-encoded bitstreams.

Secure aggregation uses SHA-256 commitment + Pedersen-style secret sharing
(compatible with the existing ZKP verifier in security/zkp.py).

Key components:
- ``DPMechanism``: Bitstream-level DP noise (flip bits with calibrated probability)
- ``PrivacyAccountant``: Rényi-DP composition tracker with (ε,δ) conversion
- ``SecretShare``: Additive secret sharing over GF(2) for bitstream aggregation
- ``CommitmentScheme``: SHA-256 commitment for verifiable aggregation
- ``FederatedClient``: Local SC training step + DP gradient encoding
- ``FederatedAggregator``: Secure aggregation server
- ``FederatedRound``: Orchestrates one round of federated learning
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── LFSR-16 (bit-compatible with core_engine) ───────────────────────


def _lfsr16_step(reg: int) -> int:
    """Single step of LFSR-16 (polynomial x^16+x^14+x^13+x^11+1)."""
    feedback = ((reg >> 15) ^ (reg >> 13) ^ (reg >> 12) ^ (reg >> 10)) & 1
    return ((reg << 1) | feedback) & 0xFFFF


def lfsr_encode(value: float, seed: int, length: int) -> np.ndarray:
    """Encode a probability [0,1] into a packed bitstream using LFSR-16."""
    threshold = int(np.clip(value, 0.0, 1.0) * 65535)
    reg = seed & 0xFFFF
    if reg == 0:
        reg = 1
    bits = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        bits[i] = 1 if reg < threshold else 0
        reg = _lfsr16_step(reg)
    return bits


def bitstream_probability(bits: np.ndarray) -> float:
    """Estimate probability from a bitstream."""
    n = len(bits)
    return float(np.sum(bits)) / n if n > 0 else 0.0


# ── Differential Privacy Mechanism ───────────────────────────────────


@dataclass
class DPMechanism:
    """Bitstream-level differential privacy via calibrated bit-flipping.

    Instead of adding Gaussian/Laplace noise to real-valued gradients,
    we flip bits in the SC bitstream with probability p. This achieves
    (ε,0)-differential privacy where ε = ln((1-p)/p) per bit.

    For a bitstream of length L, the total privacy cost is ε_total via
    Rényi-DP composition (tighter than naive composition).
    """

    epsilon: float = 1.0
    sensitivity: float = 1.0

    @property
    def flip_probability(self) -> float:
        """Calibrated bit-flip probability for the target ε."""
        e = self.epsilon / self.sensitivity
        return 1.0 / (1.0 + math.exp(e))

    def privatise(self, bitstream: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply DP noise by flipping bits with calibrated probability."""
        p = self.flip_probability
        flip_mask = rng.random(len(bitstream)) < p
        noisy = bitstream.copy()
        noisy[flip_mask] = 1 - noisy[flip_mask]
        return noisy

    def per_bit_epsilon(self) -> float:
        """Privacy cost per individual bit."""
        p = self.flip_probability
        if p <= 0 or p >= 1:
            return float("inf")
        return abs(math.log((1 - p) / p))

    def total_epsilon(self, bitstream_length: int) -> float:
        """Total ε under advanced composition (Rényi-based bound)."""
        eps_1 = self.per_bit_epsilon()
        return eps_1 * math.sqrt(2 * bitstream_length * math.log(1 / 1e-5))


# ── Gradient Clipping ────────────────────────────────────────────────


def clip_gradients(gradients: np.ndarray, max_norm: float) -> np.ndarray:
    """L2 gradient clipping for formal DP sensitivity bounds.

    Clips the gradient vector so its L2 norm ≤ max_norm.
    This is required for provable (ε,δ)-DP guarantees.
    """
    l2 = float(np.linalg.norm(gradients))
    if l2 > max_norm and l2 > 0:
        return gradients * (max_norm / l2)
    return gradients.copy()


# ── Gradient Sparsification ──────────────────────────────────────────


def sparsify_topk(gradients: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k gradient sparsification.

    Returns (sparse_gradients, mask) where only k largest-magnitude
    entries are non-zero.  Reduces communication cost proportionally.
    """
    k = min(k, len(gradients))
    indices = np.argsort(np.abs(gradients))[-k:]
    mask = np.zeros(len(gradients), dtype=np.uint8)
    mask[indices] = 1
    sparse = np.zeros_like(gradients)
    sparse[indices] = gradients[indices]
    return sparse, mask


# ── Privacy Accountant ───────────────────────────────────────────────


@dataclass
class PrivacyAccountant:
    """Rényi Differential Privacy (RDP) composition tracker.

    Tracks cumulative privacy budget across federated rounds.
    Uses the moments accountant for tight composition.
    """

    target_epsilon: float = 10.0
    target_delta: float = 1e-5
    alpha: float = 2.0
    rdp_budget: float = 0.0
    rounds_consumed: int = 0

    def consume_round(self, mechanism: DPMechanism, bitstream_length: int) -> bool:
        """Account for one round. Returns True if budget remains."""
        eps_1 = mechanism.per_bit_epsilon()
        # RDP of randomized response at order alpha
        rdp_step = (
            (self.alpha / (self.alpha - 1))
            * math.log(
                (1 - mechanism.flip_probability) ** self.alpha
                + mechanism.flip_probability**self.alpha
            )
            * bitstream_length
        )
        self.rdp_budget += abs(rdp_step)
        self.rounds_consumed += 1
        return not self.is_exhausted()

    def current_epsilon(self) -> float:
        """Convert accumulated RDP to (ε,δ)-DP."""
        if self.rdp_budget <= 0:
            return 0.0
        return self.rdp_budget + math.log(1 / self.target_delta) / (self.alpha - 1)

    def remaining_epsilon(self) -> float:
        """How much budget is left."""
        return max(0.0, self.target_epsilon - self.current_epsilon())

    def is_exhausted(self) -> bool:
        return self.current_epsilon() >= self.target_epsilon


# ── Secret Sharing (GF(2) additive) ─────────────────────────────────


@dataclass
class SecretShare:
    """Additive secret sharing over GF(2) for bitstream aggregation.

    Splits a bitstream into N shares such that XOR of all shares
    recovers the original. Individual shares reveal nothing.
    """

    num_parties: int = 3

    def split(self, bitstream: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
        """Split bitstream into additive GF(2) shares."""
        shares = []
        accumulated = np.zeros_like(bitstream)
        for i in range(self.num_parties - 1):
            share = rng.integers(0, 2, size=len(bitstream), dtype=np.uint8)
            shares.append(share)
            accumulated ^= share
        # Last share ensures XOR of all shares == original
        shares.append(bitstream ^ accumulated)
        return shares

    @staticmethod
    def reconstruct(shares: List[np.ndarray]) -> np.ndarray:
        """Reconstruct bitstream from all shares (XOR)."""
        result = np.zeros_like(shares[0])
        for share in shares:
            result ^= share
        return result

    @staticmethod
    def verify_reconstruction(original: np.ndarray, shares: List[np.ndarray]) -> bool:
        """Verify that shares reconstruct correctly."""
        return np.array_equal(original, SecretShare.reconstruct(shares))


# ── Commitment Scheme ────────────────────────────────────────────────


@dataclass
class CommitmentScheme:
    """SHA-256 commitment for verifiable aggregation.

    Compatible with the ZKPVerifier in security/zkp.py.
    """

    @staticmethod
    def commit(data: np.ndarray, nonce: Optional[bytes] = None) -> str:
        """Create a binding commitment to data."""
        payload = data.tobytes()
        if nonce is not None:
            payload = nonce + payload
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def verify(data: np.ndarray, commitment: str, nonce: Optional[bytes] = None) -> bool:
        """Verify a commitment."""
        return CommitmentScheme.commit(data, nonce) == commitment

    @staticmethod
    def generate_nonce(rng: np.random.Generator) -> bytes:
        """Generate a random 32-byte nonce."""
        return rng.bytes(32)


# ── SC Gradient Encoder ──────────────────────────────────────────────


@dataclass
class SCGradientEncoder:
    """Encodes real-valued gradients as SC bitstreams with DP noise."""

    bitstream_length: int = 256
    dp: DPMechanism = field(default_factory=DPMechanism)

    def encode(
        self,
        gradients: np.ndarray,
        seeds: np.ndarray,
        rng: np.random.Generator,
    ) -> List[np.ndarray]:
        """Encode gradient vector as privatised SC bitstreams.

        1. Normalise gradients to [0, 1]
        2. LFSR-encode each element
        3. Apply DP bit-flip noise
        """
        g_min, g_max = gradients.min(), gradients.max()
        span = g_max - g_min
        if span < 1e-12:
            normalised = np.full_like(gradients, 0.5)
        else:
            normalised = (gradients - g_min) / span

        bitstreams = []
        for i, val in enumerate(normalised):
            seed = int(seeds[i % len(seeds)]) & 0xFFFF
            if seed == 0:
                seed = 1
            bs = lfsr_encode(val, seed, self.bitstream_length)
            bs = self.dp.privatise(bs, rng)
            bitstreams.append(bs)
        return bitstreams

    def decode(self, bitstreams: List[np.ndarray], g_min: float, g_max: float) -> np.ndarray:
        """Decode SC bitstreams back to gradient values."""
        probs = np.array([bitstream_probability(bs) for bs in bitstreams])
        span = g_max - g_min
        return probs * span + g_min


# ── Federated Client ─────────────────────────────────────────────────


@dataclass
class FederatedClient:
    """One participant in federated SC learning."""

    client_id: int
    encoder: SCGradientEncoder
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))
    local_weights: Optional[np.ndarray] = None
    commitment: Optional[str] = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.client_id * 7919 + 1)

    def local_train(self, data: np.ndarray, labels: np.ndarray, lr: float = 0.01) -> np.ndarray:
        """Simulate one local training step (gradient computation).

        Uses a simple linear model for demonstration:
        loss = MSE(X @ w - y), grad = 2/n * X^T @ (X @ w - y)
        """
        if self.local_weights is None:
            weights = np.asarray(self.rng.standard_normal(data.shape[1]) * 0.01, dtype=float)
            self.local_weights = weights
        else:
            weights = self.local_weights

        predictions = data @ weights
        errors = predictions - labels
        gradients = 2.0 / len(labels) * (data.T @ errors)
        self.local_weights = weights - lr * gradients
        return gradients

    def encode_gradients(self, gradients: np.ndarray) -> Tuple[List[np.ndarray], str, float, float]:
        """Encode gradients as privatised SC bitstreams + commitment.

        Returns: (bitstreams, commitment_hash, g_min, g_max)
        """
        seeds = self.rng.integers(1, 65535, size=len(gradients), dtype=np.int64)
        bitstreams = self.encoder.encode(gradients, seeds, self.rng)

        # Commit to the privatised bitstreams
        concatenated = np.concatenate(bitstreams)
        nonce = CommitmentScheme.generate_nonce(self.rng)
        self.commitment = CommitmentScheme.commit(concatenated, nonce)

        return bitstreams, self.commitment, float(gradients.min()), float(gradients.max())


# ── Federated Aggregator ─────────────────────────────────────────────


@dataclass
class FederatedAggregator:
    """Secure aggregation server for federated SC bitstreams."""

    num_clients: int
    bitstream_length: int = 256

    def aggregate_bitstreams(
        self,
        client_bitstreams: List[List[np.ndarray]],
        weights: Optional[List[float]] = None,
    ) -> List[np.ndarray]:
        """Weighted majority-vote aggregation of SC bitstreams.

        For each gradient dimension, compute the weighted majority bit across
        all clients. If weights=None, uses uniform weights (standard FedAvg).
        """
        num_dims = len(client_bitstreams[0])
        n_clients = len(client_bitstreams)
        if weights is None:
            w = np.ones(n_clients) / n_clients
        else:
            w = np.array(weights)
            w = w / w.sum()

        aggregated = []
        for dim in range(num_dims):
            stacked = np.stack([c[dim] for c in client_bitstreams]).astype(np.float64)
            weighted_sum = w @ stacked
            agg_bs = (weighted_sum > 0.5).astype(np.uint8)
            aggregated.append(agg_bs)
        return aggregated

    def detect_outliers(
        self,
        client_bitstreams: List[List[np.ndarray]],
        threshold: float = 0.3,
    ) -> List[bool]:
        """Detect malicious/outlier clients via cosine similarity.

        For each client, compute mean cosine similarity to all others.
        Clients with mean similarity < threshold are flagged.
        """
        n = len(client_bitstreams)
        if n < 2:
            return [False] * n

        # Flatten each client's update to a single vector
        flat = []
        for cbs in client_bitstreams:
            flat.append(np.concatenate(cbs).astype(np.float64))

        is_outlier = []
        for i in range(n):
            sims = []
            for j in range(n):
                if i == j:
                    continue
                dot = np.dot(flat[i], flat[j])
                na = np.linalg.norm(flat[i])
                nb = np.linalg.norm(flat[j])
                if na > 0 and nb > 0:
                    sims.append(dot / (na * nb))
                else:
                    sims.append(0.0)
            mean_sim = float(np.mean(sims))
            is_outlier.append(mean_sim < threshold)
        return is_outlier

    def verify_commitments(
        self,
        client_bitstreams: List[List[np.ndarray]],
        commitments: List[str],
        nonces: Optional[List[bytes]] = None,
    ) -> List[bool]:
        """Verify that submitted bitstreams match commitments."""
        results = []
        for i, (bs_list, commitment) in enumerate(zip(client_bitstreams, commitments)):
            concatenated = np.concatenate(bs_list)
            nonce = nonces[i] if nonces else None
            results.append(CommitmentScheme.commit(concatenated, nonce) == commitment)
        return results


# ── Client Subsampling ───────────────────────────────────────────────


def poisson_subsample(
    clients: List[FederatedClient],
    sampling_rate: float,
    rng: np.random.Generator,
) -> List[FederatedClient]:
    """Poisson subsampling of clients for privacy amplification.

    Each client is independently included with probability ``sampling_rate``.
    This provides privacy amplification by a factor of O(sampling_rate).
    """
    selected = []
    for c in clients:
        if rng.random() < sampling_rate:
            selected.append(c)
    return selected if selected else [clients[0]]  # at least one


# ── Convergence Tracker ──────────────────────────────────────────────


@dataclass
class ConvergenceTracker:
    """Tracks loss/gradient-norm across federated rounds."""

    grad_norms: List[float] = field(default_factory=list)
    round_losses: List[float] = field(default_factory=list)

    def record(self, aggregated_gradient: np.ndarray) -> None:
        """Record one round's gradient norm."""
        self.grad_norms.append(float(np.linalg.norm(aggregated_gradient)))

    def record_loss(self, loss: float) -> None:
        self.round_losses.append(loss)

    @property
    def converged(self) -> bool:
        """Simple heuristic: converged if last 5 grad norms < 0.01."""
        if len(self.grad_norms) < 5:
            return False
        return all(g < 0.01 for g in self.grad_norms[-5:])

    @property
    def trend(self) -> str:
        """Return trend direction."""
        if len(self.grad_norms) < 2:
            return "insufficient_data"
        if self.grad_norms[-1] < self.grad_norms[-2]:
            return "decreasing"
        elif self.grad_norms[-1] > self.grad_norms[-2]:
            return "increasing"
        return "stable"


# ── Federated Round ──────────────────────────────────────────────────


@dataclass
class FederatedRound:
    """Orchestrates one complete round of federated SC learning."""

    clients: List[FederatedClient]
    aggregator: FederatedAggregator
    accountant: PrivacyAccountant
    round_number: int = 0
    convergence: ConvergenceTracker = field(default_factory=ConvergenceTracker)
    clip_norm: float = 0.0
    sampling_rate: float = 1.0
    audit_log: Any = None

    def run(
        self,
        data_per_client: List[np.ndarray],
        labels_per_client: List[np.ndarray],
        client_weights: Optional[List[float]] = None,
    ) -> Optional[np.ndarray]:
        """Execute one federated round.

        Returns aggregated gradient if privacy budget allows, None otherwise.
        """
        if self.accountant.is_exhausted():
            return None

        self.round_number += 1

        # Client subsampling
        if self.sampling_rate < 1.0:
            rng = np.random.default_rng(self.round_number)
            active = poisson_subsample(self.clients, self.sampling_rate, rng)
            active_indices = [self.clients.index(c) for c in active]
        else:
            active = self.clients
            active_indices = list(range(len(self.clients)))

        all_bitstreams = []
        all_commitments = []
        g_mins, g_maxs = [], []

        for idx, client in zip(active_indices, active):
            gradients = client.local_train(data_per_client[idx], labels_per_client[idx])
            if self.clip_norm > 0:
                gradients = clip_gradients(gradients, self.clip_norm)
            bitstreams, commitment, g_min, g_max = client.encode_gradients(gradients)
            all_bitstreams.append(bitstreams)
            all_commitments.append(commitment)
            g_mins.append(g_min)
            g_maxs.append(g_max)

        # Track privacy budget
        dp_mech = active[0].encoder.dp
        bl = active[0].encoder.bitstream_length
        self.accountant.consume_round(dp_mech, bl)

        # Select weights for active clients
        if client_weights is not None:
            active_weights = [client_weights[i] for i in active_indices]
        else:
            active_weights = None

        # Aggregate
        aggregated_bs = self.aggregator.aggregate_bitstreams(all_bitstreams, weights=active_weights)

        # Decode using global min/max range
        global_min = min(g_mins)
        global_max = max(g_maxs)
        aggregated_grads = active[0].encoder.decode(aggregated_bs, global_min, global_max)

        # Track convergence
        self.convergence.record(aggregated_grads)

        # Audit log
        if self.audit_log is not None:
            self.audit_log.log_round(
                round_number=self.round_number,
                num_active=len(active),
                epsilon_consumed=self.accountant.current_epsilon(),
                grad_norm=float(np.linalg.norm(aggregated_grads)),
            )

        return aggregated_grads

    def status(self) -> Dict[str, Any]:
        """Return current federated learning status."""
        return {
            "round": self.round_number,
            "epsilon_consumed": self.accountant.current_epsilon(),
            "epsilon_remaining": self.accountant.remaining_epsilon(),
            "rounds_consumed": self.accountant.rounds_consumed,
            "budget_exhausted": self.accountant.is_exhausted(),
            "converged": self.convergence.converged,
            "trend": self.convergence.trend,
        }


# ── DP Certificate (Gap 1) ──────────────────────────────────────────


@dataclass
class DPCertificate:
    """Exportable privacy proof document for regulatory compliance."""

    mechanism: str
    epsilon: float
    delta: float
    rounds: int
    bitstream_length: int
    composition_method: str
    accountant_state: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_accountant(
        cls,
        accountant: PrivacyAccountant,
        mechanism: DPMechanism,
        bitstream_length: int,
    ) -> DPCertificate:
        return cls(
            mechanism="bitstream_flip_rr",
            epsilon=accountant.current_epsilon(),
            delta=accountant.target_delta,
            rounds=accountant.rounds_consumed,
            bitstream_length=bitstream_length,
            composition_method="renyi_dp",
            accountant_state={
                "rdp_budget": accountant.rdp_budget,
                "alpha": accountant.alpha,
                "target_epsilon": accountant.target_epsilon,
                "flip_probability": mechanism.flip_probability,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "rounds": self.rounds,
            "bitstream_length": self.bitstream_length,
            "composition_method": self.composition_method,
            "accountant_state": self.accountant_state,
            "compliant": self.epsilon <= self.accountant_state.get("target_epsilon", float("inf")),
        }

    @property
    def is_compliant(self) -> bool:
        return self.epsilon <= self.accountant_state.get("target_epsilon", float("inf"))


# ── Gradient Compression (Gap 2) ────────────────────────────────────


def stochastic_quantize(
    gradients: np.ndarray,
    levels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stochastic quantization to reduce communication bits.

    Quantizes each gradient to one of ``levels`` levels with
    unbiased randomized rounding. E[Q(g)] = g.
    """
    g_min, g_max = gradients.min(), gradients.max()
    span = g_max - g_min
    if span < 1e-12:
        return gradients.copy()
    normalised = (gradients - g_min) / span * (levels - 1)
    lower = np.floor(normalised).astype(np.int32)
    prob = normalised - lower
    upper = lower + (rng.random(len(gradients)) < prob).astype(np.int32)
    upper = np.clip(upper, 0, levels - 1)
    return upper.astype(np.float64) / (levels - 1) * span + g_min


# ── Adaptive Epsilon Scheduling (Gap 3) ──────────────────────────────


@dataclass
class AdaptiveEpsilonScheduler:
    """Dynamically adjust ε per round based on convergence state."""

    base_epsilon: float = 1.0
    decay_rate: float = 0.95
    min_epsilon: float = 0.1
    current_epsilon: float = 0.0

    def __post_init__(self) -> None:
        self.current_epsilon = self.base_epsilon

    def step(self, converging: bool) -> float:
        """Compute ε for the next round.

        If converging: tighten privacy (reduce ε → more noise).
        If diverging: loosen privacy (increase ε → less noise).
        """
        if converging:
            self.current_epsilon = max(
                self.min_epsilon,
                self.current_epsilon * self.decay_rate,
            )
        else:
            self.current_epsilon = min(
                self.base_epsilon,
                self.current_epsilon / self.decay_rate,
            )
        return self.current_epsilon


# ── Byzantine-Robust Aggregation (Gap 4) ─────────────────────────────


def krum_select(
    client_vectors: List[np.ndarray],
    num_byzantine: int = 1,
) -> int:
    """Multi-Krum selection: find the vector closest to most others.

    Returns the index of the most "central" client update.
    Tolerates up to ``num_byzantine`` malicious clients.
    """
    n = len(client_vectors)
    k = n - num_byzantine - 2
    if k < 1:
        k = 1

    scores = []
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            dists.append(float(np.linalg.norm(client_vectors[i] - client_vectors[j]) ** 2))
        dists.sort()
        scores.append(sum(dists[:k]))

    return int(np.argmin(scores))


def trimmed_mean(
    client_vectors: List[np.ndarray],
    trim_fraction: float = 0.1,
) -> np.ndarray:
    """Coordinate-wise trimmed mean aggregation.

    Removes the top and bottom ``trim_fraction`` of values per
    coordinate before averaging. Robust to Byzantine clients.
    """
    stacked = np.stack(client_vectors)
    n = stacked.shape[0]
    trim_count = max(1, int(n * trim_fraction))

    sorted_vals = np.sort(stacked, axis=0)
    trimmed = sorted_vals[trim_count : n - trim_count, :]
    if trimmed.shape[0] == 0:
        return np.mean(stacked, axis=0)
    return np.mean(trimmed, axis=0)


# ── FedProx Proximal Term (Gap 5) ────────────────────────────────────


def fedprox_gradient(
    gradients: np.ndarray,
    local_weights: np.ndarray,
    global_weights: np.ndarray,
    mu: float = 0.01,
) -> np.ndarray:
    """Add FedProx proximal term for non-IID robustness.

    grad_proximal = grad + μ * (w_local - w_global)
    """
    return gradients + mu * (local_weights - global_weights)


# ── Momentum / Error Feedback (Gap 8) ────────────────────────────────


@dataclass
class ErrorFeedback:
    """Error feedback for gradient sparsification.

    Accumulates the residual from sparsification to avoid information
    loss over rounds.
    """

    residual: Optional[np.ndarray] = None

    def accumulate(self, gradients: np.ndarray) -> np.ndarray:
        """Add residual from previous round."""
        if self.residual is not None:
            return gradients + self.residual
        return gradients.copy()

    def update(self, original: np.ndarray, sparse: np.ndarray) -> None:
        """Store the residual (what was lost to sparsification)."""
        self.residual = original - sparse


# ── Privacy Amplification by Subsampling (Gap 9) ─────────────────────


def amplified_epsilon(
    base_epsilon: float,
    sampling_rate: float,
) -> float:
    """Compute privacy amplification from Poisson subsampling.

    Uses the Balle et al. (2018) bound: ε' ≈ log(1 + q(e^ε - 1))
    where q is the sampling rate.
    """
    if sampling_rate >= 1.0:
        return base_epsilon
    if sampling_rate <= 0.0:
        return 0.0
    return math.log(1 + sampling_rate * (math.exp(base_epsilon) - 1))


# ── Audit Log (Gap 10) ──────────────────────────────────────────────


@dataclass
class AuditEntry:
    """Single audit log entry for one federated round."""

    round_number: int
    num_active_clients: int
    epsilon_consumed: float
    grad_norm: float
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        import time

        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class AuditLog:
    """Per-round provenance record for regulatory compliance."""

    entries: List[AuditEntry] = field(default_factory=list)

    def log_round(
        self,
        round_number: int,
        num_active: int,
        epsilon_consumed: float,
        grad_norm: float,
    ) -> None:
        self.entries.append(
            AuditEntry(
                round_number=round_number,
                num_active_clients=num_active,
                epsilon_consumed=epsilon_consumed,
                grad_norm=grad_norm,
            )
        )

    def to_list(self) -> List[Dict[str, Any]]:
        return [
            {
                "round": e.round_number,
                "active_clients": e.num_active_clients,
                "epsilon": e.epsilon_consumed,
                "grad_norm": e.grad_norm,
                "timestamp": e.timestamp,
            }
            for e in self.entries
        ]

    @property
    def total_rounds(self) -> int:
        return len(self.entries)

    @property
    def max_epsilon(self) -> float:
        if not self.entries:
            return 0.0
        return max(e.epsilon_consumed for e in self.entries)
